
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import nibabel as nib

from .dataset import MultiOrientationImages
from .network import FusionNet
from .utils import mkdir_p

class FusionNetwork:
    def __init__(self, pretrained_model=None, gpu=0):
        self.pretrained_model = pretrained_model
        self.device = torch.device('cuda:0' if gpu == 0 else 'cuda:1')

        # define network
        self.fusion_net = FusionNet(in_ch=3, out_ch=1)

        # initialize training variables
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.batch_size = None
        self.optim_fusion_net = None

        # pretrained model
        self.checkpoint = None
        if self.pretrained_model is not None:
            self.checkpoint = torch.load(self.pretrained_model, map_location=self.device)
            self.fusion_net.load_state_dict(self.checkpoint['fusion_net'])

        # send to device
        self.fusion_net.to(self.device)
        self.start_epoch = 0

    def load_dataset(self, dataset_dir, data_name, batch_size):
        train_dataset = MultiOrientationImages(dataset_dir, data_name, 'train')
        valid_dataset = MultiOrientationImages(dataset_dir, data_name, 'valid')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    def initialize_training(self, out_dir, lr):
        self.out_dir = out_dir
        mkdir_p(self.out_dir)
        mkdir_p(os.path.join(out_dir, 'results'))
        mkdir_p(os.path.join(out_dir, 'models'))

        # define losses
        self.l1_loss = nn.L1Loss(reduction='none')

        # define optimizers
        self.optim_fusion_net = torch.optim.AdamW(self.fusion_net.parameters(), lr=lr)
        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optim_fusion_net.load_state_dict(self.checkpoint['optim_fusion_net'])
        self.start_epoch += 1

    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs+1):
            self.train_loader = tqdm(self.train_loader)
            self.fusion_net.train()
            train_loss_sum = 0.0
            num_train_imgs = 0

            for batch_id, (imgs, ori_img) in enumerate(self.train_loader):
                imgs = tuple([img.to(self.device) for img in imgs])
                ori_img = ori_img.to(self.device)
                curr_batch_size = ori_img.size()[0]
                imgs = torch.cat(imgs, dim=1)
                syn_img = self.fusion_net(imgs)

                loss = self.cal_loss(syn_img, ori_img)
                train_loss_sum += loss * curr_batch_size
                num_train_imgs += curr_batch_size
                self.train_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {loss:.3f}; '
                                                   f'avg_train: {train_loss_sum/num_train_imgs:.3f}; '))

                if batch_id % 40 == 0:
                    img_affine = [[-1, 0, 0, 96], [0, -1, 0, 96], [0, 0, 1, -78], [0, 0, 0, 1]]
                    img_save = np.array(syn_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                    img_save = nib.Nifti1Image(img_save, img_affine)
                    file_name = os.path.join(self.out_dir, 'results', f'train_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_syn.nii.gz')
                    nib.save(img_save, file_name)

                    img_save = np.array(ori_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                    img_save = nib.Nifti1Image(img_save, img_affine)
                    file_name = os.path.join(self.out_dir, 'results', f'train_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_ori.nii.gz')
                    nib.save(img_save, file_name)

            if epoch % 1 == 0:
                file_name = os.path.join(self.out_dir, 'models', f'epoch{str(epoch).zfill(3)}.pt')
                self.save_model(file_name, epoch)

            # Clear GPU memory to avoid OOM
            torch.cuda.empty_cache()

            # validation
            self.valid_loader = tqdm(self.valid_loader)
            valid_loss_sum = 0.0
            num_valid_imgs = 0
            self.fusion_net.eval()
            with torch.set_grad_enabled(False):
                for batch_id, (imgs, ori_img) in enumerate(self.valid_loader):
                    imgs = tuple([img.to(self.device) for img in imgs])
                    ori_img = ori_img.to(self.device)
                    curr_batch_size = ori_img.size()[0]
                    imgs = torch.cat(imgs, dim=1)
                    syn_img = self.fusion_net(imgs)

                    loss = self.cal_loss(syn_img, ori_img, is_train=False)
                    valid_loss_sum += loss * curr_batch_size
                    num_valid_imgs += curr_batch_size
                    self.valid_loader.set_description((f'epoch: {epoch}; '
                                                       f'rec: {loss:.3f}; '
                                                       f'avg_valid: {valid_loss_sum/num_valid_imgs:.3f}; '))

                    if batch_id == 0:
                        img_affine = [[-1, 0, 0, 96], [0, -1, 0, 96], [0, 0, 1, -78], [0, 0, 0, 1]]
                        img_save = np.array(syn_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                        img_save = nib.Nifti1Image(img_save, img_affine)
                        file_name = os.path.join(self.out_dir, 'results', f'valid_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_syn.nii.gz')
                        nib.save(img_save, file_name)

                        img_save = np.array(ori_img.detach().cpu().squeeze().permute(1,2,0).permute(1,0,2))
                        img_save = nib.Nifti1Image(img_save, img_affine)
                        file_name = os.path.join(self.out_dir, 'results', f'valid_epoch{str(epoch).zfill(2)}_batch{str(batch_id).zfill(3)}_ori.nii.gz')
                        nib.save(img_save, file_name)

    def test(self, imgs, out_dir, prefix, img_affine, img_hdr, norm=1000):
        self.fusion_net.eval()
        with torch.set_grad_enabled(False):
            imgs = tuple([img.to(self.device) for img in imgs])
            imgs = torch.cat(imgs, dim=1)
            fuse_img = self.fusion_net(imgs)

            img_save = np.array(fuse_img.cpu().squeeze().permute(1,2,0).permute(1,0,2))
            img_save = img_save / 0.25 * norm
            img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
            file_name = os.path.join(out_dir, f'{prefix}_fusion.nii.gz')
            nib.save(img_save, file_name)

    def cal_loss(self, syn_img, ori_img, is_train=True):
        rec_loss = self.l1_loss(syn_img, ori_img).mean()

        if is_train:
            self.optim_fusion_net.zero_grad()
            rec_loss.backward()
            self.optim_fusion_net.step()
        return rec_loss.item()

    def save_model(self, file_name, epoch):
        state = {
            'epoch': epoch,
            'fusion_net': self.fusion_net.state_dict(),
            'optim_fusion_net': self.optim_fusion_net.state_dict()
        }
        torch.save(obj=state, f=file_name)
