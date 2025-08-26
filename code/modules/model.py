#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import csv
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import utils
import torch.nn.functional as F

from .dataset import PairedMRI
from .network import Unet, ThetaEncoder, Discriminator, Vgg16, DomainAdaptorBeta, DomainAdaptorTheta
from .utils import mkdir_p, create_one_hot, softmax, KLDivergenceLoss, TemperatureAnneal

import nibabel as nib

class CALAMITI:
    def __init__(self, beta_dim, theta_dim, train_sample='st_gumbel_softmax', valid_sample='argmax',
                 pretrained_model=None, initial_temp=1.0, anneal_rate=5e-4, gpu=0, fine_tune=False):
        self.beta_dim = beta_dim
        self.theta_dim = theta_dim
        self.train_sample = train_sample
        self.valid_sample = valid_sample
        self.initial_temp = initial_temp
        self.anneal_rate = anneal_rate
        self.device = torch.device('cuda:0' if gpu==0 else 'cuda:1')
        self.fine_tune = fine_tune if pretrained_model is not None else False

        if self.fine_tune:
            print('Fine tuning network...')

        # define networks
        self.beta_encoder = Unet(in_ch=1, out_ch=16, num_lvs=4, base_ch=8, final_act='noact')
        self.da_beta = DomainAdaptorBeta(in_ch=16, out_ch=self.beta_dim, final_act=False)
        self.theta_encoder = ThetaEncoder(in_ch=1, out_ch=128)
        self.da_theta = DomainAdaptorTheta(out_ch=self.theta_dim)
        self.decoder = Unet(in_ch=self.theta_dim+self.beta_dim, num_lvs=4, base_ch=16, out_ch=1, final_act='noact')
        self.discriminator = Discriminator(in_ch=self.beta_dim+1, out_ch=1)
        self.vgg = Vgg16(requires_grad=False)

        # initialize training variables
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.batch_size = None
        self.optim_beta_encoder, self.optim_theta_encoder, self.optim_decoder = None, None, None
        self.optim_discriminator = None
        self.optim_da_beta, self.optim_da_theta = None, None
        self.temp_sched = None

        # pretrained models
        self.checkpoint = None
        if pretrained_model is not None:
            self.checkpoint = torch.load(pretrained_model, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.theta_encoder.load_state_dict(self.checkpoint['theta_encoder'])
            self.decoder.load_state_dict(self.checkpoint['decoder'])
            self.discriminator.load_state_dict(self.checkpoint['discriminator'])
            self.da_beta.load_state_dict(self.checkpoint['da_beta'])
            self.da_theta.load_state_dict(self.checkpoint['da_theta'])

        # send to device
        self.beta_encoder.to(self.device)
        self.theta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)
        self.vgg.to(self.device)
        self.da_beta.to(self.device)
        self.da_theta.to(self.device)
        self.start_epoch = 0

    def load_dataset(self, dataset_dirs, data_names, orientations, batch_size):
        train_dataset = PairedMRI(dataset_dirs, data_names, orientations, 'train')
        valid_dataset = PairedMRI(dataset_dirs, data_names, orientations, 'valid')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def initialize_training(self, out_dir, lr):
        self.out_dir = out_dir
        mkdir_p(self.out_dir)
        mkdir_p(os.path.join(out_dir, 'results'))
        mkdir_p(os.path.join(out_dir, 'models'))
        self.writer = SummaryWriter(os.path.join(out_dir, 'results'))

        # define loss
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = KLDivergenceLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        self.optim_beta_encoder = torch.optim.Adam(self.beta_encoder.parameters(), lr=lr)
        self.optim_theta_encoder = torch.optim.Adam(self.theta_encoder.parameters(), lr=lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optim_da_beta = torch.optim.Adam(self.da_beta.parameters(), lr=lr)
        self.optim_da_theta = torch.optim.Adam(self.da_theta.parameters(), lr=lr)
        self.temp_sched = TemperatureAnneal(initial_temp=self.initial_temp, anneal_rate=self.anneal_rate, min_temp=0.5, device=self.device)

        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optim_beta_encoder.load_state_dict(self.checkpoint['optim_beta_encoder'])
            self.optim_theta_encoder.load_state_dict(self.checkpoint['optim_theta_encoder'])
            self.optim_decoder.load_state_dict(self.checkpoint['optim_decoder'])
            self.optim_discriminator.load_state_dict(self.checkpoint['optim_discriminator'])
            self.temp_sched.load_state_dict(self.checkpoint['temp_sched'])
            self.optim_da_beta.load_state_dict(self.checkpoint['optim_da_beta'])
            self.optim_da_theta.load_state_dict(self.checkpoint['optim_da_theta'])
        self.start_epoch += 1

    def reparameterize_logit(self, logit, method):
        tau = self.temp_sched.get_temp() if self.temp_sched is not None else 0.5
        if method == 'gumbel_softmax':
            beta = F.gumbel_softmax(logit, tau=tau, dim=1, hard=False)
        elif method == 'st_gumbel_softmax':
            beta = F.gumbel_softmax(logit, tau=tau, dim=1, hard=True)
        elif method == 'argmax':
            beta = create_one_hot(logit, dim=1)
        else:
            beta = softmax(logit, self.temp_sched.get_temp(), dim=1)
        return beta

    def cal_beta(self, imgs, method):
        logits = []
        betas = []
        for img in imgs:
            logit_map = self.beta_encoder(img)
            logit = self.da_beta(logit_map)
            #beta = logit
            beta = self.reparameterize_logit(logit, method)
            logits.append(logit)
            betas.append(beta)
        return tuple(betas), tuple(logits)

    def cal_theta(self, imgs):
        thetas = []
        mus = []
        logvars = []
        for img in imgs:
            theta, mu, logvar = self.da_theta(self.theta_encoder(img), self.device)
            thetas.append(theta)
            mus.append(mu)
            logvars.append(logvar)
        return thetas, mus, logvars

    def decode(self, betas, thetas, swap_beta=True, self_rec=False):
        rec_imgs = []
        img_ids = []  # which modality is used during decoding
        beta_combined = torch.cat(betas, dim=1)
        num_modalities = len(betas)
        for img_id, theta in enumerate(thetas):
            if swap_beta:
                beta_id = [np.random.randint(num_modalities) * self.beta_dim + i for i in range(self.beta_dim)]
                beta = beta_combined[:, beta_id, :, :]
                combined_img = torch.cat([beta, theta.repeat(1,1,beta.shape[2],beta.shape[3])], dim=1)
                rec_img = self.decoder(combined_img)
                rec_imgs.append(rec_img)
                img_ids.append(img_id)
            else:
                for modality_id in range(num_modalities):
                    beta_id = [modality_id * self.beta_dim + i for i in range(self.beta_dim)]
                    beta = beta_combined[:, beta_id, :, :]
                    combined_img = torch.cat([beta, theta.repeat(1,1,beta.shape[2],beta.shape[3])], dim=1)
                    rec_img = self.decoder(combined_img)
                    if self_rec:
                        img_ids.append(img_id)
                        rec_imgs.append(rec_img)
                    else:
                        if modality_id != img_id:
                            img_ids.append(img_id)
                            rec_imgs.append(rec_img)
        return tuple(rec_imgs), tuple(img_ids)    

    def cal_loss(self, betas, thetas, dataset_id, rec_imgs=None, imgs=None, img_ids=None,
                 mus=None, logvars=None, is_train=True, fine_tune=False):
        rec_loss = 0.0
        kld_loss = 0.0
        gen_loss = 0.0
        dis_loss = 0.0
        per_loss = 0.0

        if rec_imgs:
            # 1. reconstruction loss (l1 and perceptual loss)
            for rec_img, img_id in zip(rec_imgs, img_ids):
                compare_img = imgs[img_id]
                # l1 loss
                rec_loss += self.l1_loss(rec_img, compare_img).mean()
            
                # perceptual loss
                rec_feature = self.vgg(rec_img.repeat(1,3,1,1)).relu2_2
                tar_feature = self.vgg(compare_img.repeat(1,3,1,1)).relu2_2
                per_loss += self.l1_loss(rec_feature, tar_feature).mean()
            per_loss = per_loss / len(rec_imgs)
            rec_loss = rec_loss / len(rec_imgs)
        
            # 2. theta kld loss
            for mu, logvar in zip(mus, logvars):
                kld_loss += self.kld_loss(mu, logvar).mean()
            kld_loss = kld_loss / len(mus)
          
            # 3. beta similarity loss
            beta_loss = self.l1_loss(betas[0], betas[1]).mean()

            # 4. generator loss loss
            curr_batch_size = betas[0].shape[0] * 2
            site_label = torch.cat(dataset_id, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            site_label.requires_grad = False
            positive_label = Variable(torch.FloatTensor(torch.ones(curr_batch_size, 1, 36, 36).float()), requires_grad=False).to(self.device)
            negative_label = Variable(torch.FloatTensor(torch.zeros(curr_batch_size, 1, 36, 36).float()), requires_grad=False).to(self.device)
            beta_combined = torch.cat(betas, dim=0)
            beta_shuffle = torch.cat([beta_combined[1:,...], beta_combined[[0],...]], dim=0)
            pred_joint = self.discriminator(torch.cat([beta_combined, site_label.repeat(1,1,288,288)], dim=1))
            pred_marginal = self.discriminator(torch.cat([beta_shuffle, site_label.repeat(1,1,288,288)], dim=1))
            gen_loss += 1.0 * (self.bce_loss(pred_joint, negative_label).mean() + self.bce_loss(pred_marginal, positive_label).mean())

            # COMBINE LOSSES
            total_loss = 4*rec_loss + 3e-2*beta_loss + 1e-7*kld_loss + 3e-2*gen_loss + 4e-2*per_loss
            if is_train:
                if not self.fine_tune:
                    self.optim_beta_encoder.zero_grad()
                    self.optim_theta_encoder.zero_grad()
                    self.optim_decoder.zero_grad()
                    self.optim_da_beta.zero_grad()
                    self.optim_da_theta.zero_grad()
                    total_loss.backward()
                    self.optim_discriminator.zero_grad()
                    self.optim_beta_encoder.step()
                    self.optim_theta_encoder.step()
                    self.optim_decoder.step()
                    self.optim_da_beta.step()
                    self.optim_da_theta.step()
                    self.temp_sched.step()

            loss = {'rec_loss': rec_loss.item(),
                    'beta_loss': beta_loss.item(),
                    'kld_loss': kld_loss.item(),
                    'gen_loss': gen_loss.item(),
                    'per_loss': per_loss.item(),
                    'total_loss': total_loss.item()}
        else:
            # 5. discriminator loss
            curr_batch_size = betas[0].shape[0] * 2
            site_label = torch.cat(dataset_id, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            site_label.requires_grad = False
            positive_label = Variable(torch.FloatTensor(torch.ones(curr_batch_size, 1, 36, 36).float()), requires_grad=False).to(self.device)
            negative_label = Variable(torch.FloatTensor(torch.zeros(curr_batch_size, 1, 36, 36).float()), requires_grad=False).to(self.device)
            beta_combined = torch.cat(betas, dim=0)
            beta_shuffle = torch.cat([beta_combined[1:,...], beta_combined[[0],...]], dim=0)
            pred_joint = self.discriminator(torch.cat([beta_combined.detach(), site_label.detach().repeat(1,1,288,288)], dim=1))
            pred_marginal = self.discriminator(torch.cat([beta_shuffle.detach(), site_label.detach().repeat(1,1,288,288)], dim=1))
            dis_loss += 1.0*(self.bce_loss(pred_joint, positive_label).mean() + self.bce_loss(pred_marginal, negative_label).mean())

            if is_train and not self.fine_tune:
                self.optim_discriminator.zero_grad()
                (3e-2 * dis_loss).backward()
                self.optim_beta_encoder.zero_grad()
                self.optim_theta_encoder.zero_grad()
                self.optim_da_theta.zero_grad()
                self.optim_da_beta.zero_grad()
                self.optim_decoder.zero_grad()
                self.optim_discriminator.step()

            loss = {'dis_loss': dis_loss.item()}
        
        return loss
                                   
    def train(self, epochs, shuffle_theta=True):
        for epoch in range(self.start_epoch, epochs+1):
            self.train_loader = tqdm(self.train_loader)
            self.beta_encoder.train()
            self.theta_encoder.train()
            self.decoder.train()
            self.discriminator.train()
            self.da_beta.train()
            self.da_theta.train()
            train_loss_sum = 0.0
            num_train_imgs = 0
            shuffle_theta = True
            for batch_id, (imgs, dataset_ids, other_imgs) in enumerate(self.train_loader):
                dataset_id = tuple([torch.FloatTensor(dataset_id_tmp.float()).to(self.device) for dataset_id_tmp in dataset_ids])
                imgs = tuple([img.to(self.device) for img in imgs])
                other_imgs = tuple([other_img.to(self.device) for other_img in other_imgs])
                curr_batch_size = imgs[0].size()[0]
                betas, logits = self.cal_beta(imgs, self.train_sample)
                if shuffle_theta:
                    thetas, mus, logvars = self.cal_theta(other_imgs)
                else:
                    thetas, mus, logvars = self.cal_theta(imgs)
                if batch_id % 2 == 0:
                    # do not decode, only update discriminator network
                    loss_dis = self.cal_loss(betas, thetas, dataset_id, is_train=True)
                else:
                    # decode beta and theta and update non-discriminator network
                    rec_imgs, img_ids = self.decode(betas, thetas, swap_beta=True, self_rec=False)
                    loss = self.cal_loss(betas, thetas, dataset_id, rec_imgs, imgs, img_ids, mus, logvars, is_train=True)
                
                    train_loss_sum += loss['total_loss'] * curr_batch_size
                    num_train_imgs += curr_batch_size
                    self.train_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {loss["rec_loss"]:.3f}; '
                                                       f'per: {loss["per_loss"]:.3f}; '
                                                       f'beta: {loss["beta_loss"]:.3f}; '
                                                       f'kld: {loss["kld_loss"]:.3f}; '
                                                       f'gen: {loss["gen_loss"]:.3f}; '
                                                       f'dis: {loss_dis["dis_loss"]:.3f}; '
                                                       f'avg_train: {train_loss_sum/num_train_imgs:.3f}; '
                                                       f'temp: {self.temp_sched.get_temp():.4f}; '))
                # 5. save training images
                if batch_id % 31 == 0 and batch_id % 2:
                    # image and rec_images
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_image'
                    self.save_image(tuple([imgs[img_id] for img_id in img_ids])+rec_imgs, file_prefix)
                    # save logits and betas
                    file_prefix = f'train_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_beta'
                    self.save_image(logits+betas, file_prefix)
                # 6. save model
                if batch_id % 1000 == 0:
                    file_name = os.path.join(self.out_dir, 'models',
                                             f'epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}.pt')
                    self.save_model(file_name, epoch)

                if batch_id % 2:
                    # 7. visualize losses using tensorboard
                    self.writer.add_scalar('training/reconstruction error',
                                           loss['rec_loss'],
                                           (epoch-1) * len(self.train_loader) + batch_id)
                    self.writer.add_scalar('training/perceptual loss',
                                           loss['per_loss'],
                                           (epoch-1)  * len(self.train_loader) + batch_id)
                    self.writer.add_scalar('training/beta similarity loss',
                                           loss['beta_loss'],
                                           (epoch-1)  * len(self.train_loader) + batch_id)
                    self.writer.add_scalar('training/kld loss',
                                           loss['kld_loss'],
                                           (epoch-1)  * len(self.train_loader) + batch_id)
                    self.writer.add_scalar('training/gen loss',
                                           loss['gen_loss'],
                                           (epoch-1)  * len(self.train_loader) + batch_id)
                    self.writer.add_scalar('training/dis loss',
                                           loss_dis['dis_loss'],
                                           (epoch-1)  * len(self.train_loader) + batch_id)
            # validation
            self.valid_loader = tqdm(self.valid_loader)
            valid_loss_sum = 0.0
            num_valid_imgs = 0
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            self.da_beta.eval()
            self.da_theta.eval()

            with torch.set_grad_enabled(False):
                for batch_id, (imgs, dataset_ids, other_imgs) in enumerate(self.valid_loader):
                    dataset_id = tuple([torch.FloatTensor(dataset_id_tmp.float()).to(self.device) for dataset_id_tmp in dataset_ids])
                    imgs = tuple([img.to(self.device) for img in imgs])
                    other_imgs = tuple([other_img.to(self.device) for other_img in other_imgs])
                    curr_batch_size = imgs[0].size()[0]
                    betas, logits = self.cal_beta(imgs, self.valid_sample)
                    thetas, mus, logvars = self.cal_theta(other_imgs)
                    rec_imgs, img_ids = self.decode(betas, mus, swap_beta=False, self_rec=True)
                    loss = self.cal_loss(betas, thetas, dataset_id, rec_imgs, imgs, img_ids, mus, logvars, is_train=False)
                    loss_dis = self.cal_loss(betas, thetas, dataset_id, is_train=False)
                    valid_loss_sum += loss['total_loss'] * curr_batch_size
                    num_valid_imgs += curr_batch_size
                    self.valid_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {loss["rec_loss"]:.3f}; '
                                                       f'per: {loss["per_loss"]:.3f}; '
                                                       f'beta: {loss["beta_loss"]:.3f}; '
                                                       f'kld: {loss["kld_loss"]:.3f}; '
                                                       f'gen: {loss["gen_loss"]:.3f}; '
                                                       f'dis: {loss_dis["dis_loss"]:.3f}; '
                                                       f'avg_valid: {valid_loss_sum/num_valid_imgs:.3f}; '
                                                       f'temp: {self.temp_sched.get_temp():.4f}; '))
                    # save validation images
                    if batch_id == 0:
                        # image and rec images
                        file_prefix = f'valid_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_image'
                        self.save_image(imgs+rec_imgs, file_prefix)
                        # save logits and betas
                        file_prefix = f'valid_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(3)}_beta'
                        self.save_image(logits+betas, file_prefix)
                    # save thetas
                    theta_save = [theta[:,:,0,0].cpu().numpy().round(5) for theta in thetas]
                    for mu in mus:
                        theta_save.append(mu[:,:,0,0].cpu().numpy().round(5))
                    for logvar in logvars:
                        theta_save.append(torch.sqrt(torch.exp(logvar[:,:,0,0])).cpu().numpy().round(5))
                    file_name = os.path.join(self.out_dir, 'results', 'theta_valid.csv')
                    self.save_theta(file_name, epoch, theta_save)

                    # 7. visualize losses using tensorboard
                    self.writer.add_scalar('validation/reconstruction error',
                                           loss['rec_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    self.writer.add_scalar('validation/perceptual loss',
                                           loss['per_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    self.writer.add_scalar('validation/beta similarity loss',
                                           loss['beta_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    self.writer.add_scalar('validation/kld loss',
                                           loss['kld_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    self.writer.add_scalar('validation/gen loss',
                                           loss['gen_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    self.writer.add_scalar('validation/dis loss',
                                           loss_dis['dis_loss'],
                                           (epoch-1)  * len(self.valid_loader) + batch_id)
                    
    def save_model(self, file_name, epoch):
        state = {'epoch': epoch,
                 'beta_encoder': self.beta_encoder.state_dict(),
                 'theta_encoder': self.theta_encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'discriminator': self.discriminator.state_dict(),
                 'da_beta': self.da_beta.state_dict(),
                 'da_theta': self.da_theta.state_dict(),
                 'optim_beta_encoder': self.optim_beta_encoder.state_dict(),
                 'optim_theta_encoder': self.optim_theta_encoder.state_dict(),
                 'optim_decoder': self.optim_decoder.state_dict(),
                 'optim_discriminator': self.optim_discriminator.state_dict(),
                 'optim_da_beta': self.optim_da_beta.state_dict(),
                 'optim_da_theta': self.optim_da_theta.state_dict(),
                 'temp_sched': self.temp_sched.state_dict()}
        torch.save(obj=state, f=file_name)


    def save_theta(self, file_name, epoch, thetas):
        theta_sz = thetas[0].shape
        if not os.path.isfile(file_name):
            with open(file_name, 'w') as fp:
                wr = csv.writer(fp)
                head = ['epoch']
                for i in range(len(thetas)):
                    for j in range(theta_sz[1]):
                        head.append('theta%d_%d' % (i,j))
                wr.writerow(head)
        
        with open(file_name, 'a') as fp:
            wr = csv.writer(fp)
            for r in range(theta_sz[0]):
                out = [str(epoch)]
                for i in range(len(thetas)):
                    for j in range(theta_sz[1]):
                        out.append(str(thetas[i][r, j]))
                wr.writerow(out)

    def save_image(self, imgs, file_prefix):
        num_modalities = len(imgs)
        num_channels = imgs[0].size()[1]
        for ch in range(num_channels):
            img_save = torch.cat([img[:4,[ch],:,:].cpu() for img in imgs], dim=0)
            grid = utils.make_grid(tensor=img_save, nrow=4, normalize=False, range=(0,1))
            if num_channels > 1:
                file_name = os.path.join(self.out_dir, 'results', file_prefix+
                                         f'_channel{str(ch).zfill(1)}.png')
            else:
                file_name = os.path.join(self.out_dir, 'results', file_prefix+'.png')
            utils.save_image(grid, file_name)
    
                
    def encode_single_img(self, img, img_slc, out_dir, prefix, orientation='axial', volume=False,
                          img_hdr=None, img_affine=None):
        mkdir_p(out_dir)
        with torch.set_grad_enabled(False):
            self.da_beta.eval()
            self.da_theta.eval()
            self.theta_encoder.eval()
            self.beta_encoder.eval()
            betas = []
            # calculate beta
            if volume:
                imgs = []
                num_slc = img.shape[0]
                for slc in range(num_slc):
                    imgs.append(img[[slc],...])
                for img in imgs:
                    img = img.to(self.device).unsqueeze(1) # generate 4d tensor
                    logit = self.da_beta(self.beta_encoder(img))
                    betas.append(self.reparameterize_logit(logit, self.valid_sample).cpu())
                beta = torch.cat(betas, dim=0)
            else:
                img = img.to(self.device).unsqueeze(1)
                logit = self.da_beta(self.beta_encoder(img))
                beta = self.reparameterize_logit(logit, self.valid_sample).cpu()
            if isinstance(img_slc, list):
                mus = []
                for slc in img_slc:     
                    slc = slc.to(self.device)
                    _, mu, _ = self.da_theta(self.theta_encoder(slc), self.device)
                    mus.append(mu)
                mu = torch.mean(torch.stack(mus, dim=0), dim=0)
                
            else:
                img_slc = img_slc.unsqueeze(1).to(self.device)
                _, mu, _ = self.da_theta(self.theta_encoder(img_slc), self.device)
            # save original image
            if not volume:
                img_save = np.array(img.cpu().squeeze().permute(1,0))
                img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
                file_name = os.path.join(out_dir, f'{prefix}.nii.gz')
                nib.save(img_save, file_name)
            else:
                if orientation == 'axial':
                    img = torch.cat(imgs, dim=0).cpu()
                    img_save = np.array(img.cpu().squeeze().permute(1,2,0).permute(1,0,2)).astype(np.float32)
                    img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
                    file_name = os.path.join(out_dir, f'{prefix}_ori.nii.gz')
                    nib.save(img_save, file_name)
            # save beta image
            if not volume:
                for ch in range(self.beta_dim):
                    img_save = nib.Nifti1Image(np.array(beta[:,[ch],:,:].cpu()).transpose(3,1,2,0).squeeze(), np.diag([1,1,1,1]))
                    file_name = os.path.join(out_dir, f'{prefix}_beta_channel{str(ch).zfill(1)}.nii.gz')
                    nib.save(img_save, file_name)
            else: # save 4D beta volume
                if orientation == 'axial':
                    img_save = np.array(beta.cpu().squeeze().permute(2,3,0,1).permute(1,0,2,3))
                elif orientation == 'coronal':
                    img_save = np.array(beta.cpu().squeeze().permute(0,3,2,1).flip(2).permute(1,0,2,3))
                elif orientation == 'sagittal':
                    img_save = np.array(beta.cpu().squeeze().permute(3,0,2,1).flip(2).permute(1,0,2,3))
                img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
                file_name = os.path.join(out_dir, f'{prefix}_beta_{orientation}.nii.gz')
                nib.save(img_save, file_name)
            if not volume:
                # save theta
                file_name = os.path.join(out_dir, prefix+'_theta.txt')
                np.savetxt(file_name, mu[:,:,0,0].cpu().numpy(), delimiter=',', fmt='%5f')
            else:
                if orientation == 'axial':
                    # save mean theta over all slices
                    file_name = os.path.join(out_dir, f'{prefix}_theta.txt')
                    theta_mean = np.mean(mu[:,:,0,0].cpu().numpy(), axis=0)
                    np.savetxt(file_name, np.expand_dims(theta_mean, axis=0), delimiter=',', fmt='%5f')

    def decode_single_img(self, beta, theta, out_dir, prefix,
                          orientation='axial', volume=False,
                          img_hdr=None, img_affine=None):
        mkdir_p(out_dir)
        with torch.set_grad_enabled(False):
            self.decoder.eval()
            self.theta_encoder.eval()
            self.beta_encoder.eval()
            self.discriminator.eval()
            theta = theta.to(self.device)
            if volume:
                # devide beta into batches
                batch_size = beta.shape[0]
                betas = []
                rec_imgs = []
                for slc in range(batch_size):
                    betas.append(beta[[slc],...])
                for beta in betas:
                    beta = beta.to(self.device)
                    combined_map = torch.cat([beta, theta.repeat(beta.size()[0],1,\
                                                    beta.size()[2],beta.size()[3])],dim=1)
                    rec_imgs.append(self.decoder(combined_map).cpu())
                rec_img = torch.cat(rec_imgs, dim=0)
            else:
                beta = beta.to(self.device)
                combined_map = torch.cat([beta, theta.repeat(beta.size()[0],1,\
                                                    beta.size()[2],beta.size()[3])],dim=1)
                rec_img = self.decoder(combined_map).cpu()
            if not volume:
                img_save = np.array(rec_img.cpu().squeeze().permute(1,0))
                img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
                file_name = os.path.join(out_dir, f'{prefix}_harmonized.nii.gz')
                nib.save(img_save, file_name)
            else:
                # save image
                if orientation == 'axial':
                    img_save = np.array(rec_img.squeeze().permute(1,2,0).permute(1,0,2))
                elif orientation == 'coronal':
                    img_save = np.array(rec_img.squeeze().permute(0,2,1).flip(2).permute(1,0,2))
                elif orientation == 'sagittal':
                    img_save = np.array(rec_img.squeeze().permute(2,0,1).flip(2).permute(1,0,2))
                img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
                file_name = os.path.join(out_dir, f'{prefix}_{orientation}_recon.nii.gz')
                nib.save(img_save, file_name)
