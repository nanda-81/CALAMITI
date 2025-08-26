#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
import nibabel as nib
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
   
from modules.model import CALAMITI

def combine_imgs(img_dirs, out_dir, prefix):
    # obtain images
    imgs = []
    for img_dir in img_dirs:
        img_file = nib.load(img_dir)
        img = torch.from_numpy(img_file.get_fdata().astype(np.float32))
        img_hdr = img_file.header
        img_affine = img_file.affine
        imgs.append(img.numpy())

    # calculate median
    img_cat = np.stack(imgs, axis=-1) 
    img_median = np.median(img_cat, axis=-1)
    #img_cat = torch.cat(imgs, dim=-1)
    #img_median = torch.median(img_cat, dim=-1).values

    # save median image
    img_save = img_median[144-120:144+121, 144-143:144+143, 144-120:144+121] * 4000.0
    img_save = nib.Nifti1Image(img_save, img_affine, img_hdr)
    file_name = os.path.join(out_dir, f'{prefix}_med_fusion.nii.gz')
    nib.save(img_save, file_name)

def load_beta(beta_dir):
    beta_file = nib.load(beta_dir)
    beta = np.array(beta_file.get_fdata().astype(np.float32))
    num_ch = beta.shape[3]
    beta_vol = []
    for ch in range(num_ch):
        beta_ch = ToTensor()(beta[:,:,:,ch])
        beta_vol.append(beta_ch.permute(2,1,0).unsqueeze(1))
    beta = torch.cat(beta_vol, dim=1)
    img_hdr = beta_file.header
    img_affine = beta_file.affine
    return beta, img_hdr, img_affine

def load_theta(theta_dir):
    try:
        theta = torch.FloatTensor([float(value) for value in theta_dir])
    except ValueError:
        theta = torch.FloatTensor(np.loadtxt(theta_dir, delimiter=',')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return theta

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='DisentangledVAE')
    parser.add_argument('--in-beta', type=str, nargs='+', required=True)
    parser.add_argument('--in-theta', type=str,  required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='sub1')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args(args)

    # initialize model
    decoder = CALAMITI(beta_dim = args.beta_dim,
                       theta_dim = args.theta_dim,
                       train_sample = 'st_gumbel_softmax',
                       valid_sample = 'argmax',
                       pretrained_model = args.pretrained_model,
                       gpu = args.gpu)
    # load data
    theta = load_theta(theta_dir = args.in_theta)
    orientations = ['axial', 'coronal', 'sagittal']
    for beta_dir, orientation in zip(args.in_beta,orientations):
        beta, img_hdr, img_affine = load_beta(beta_dir)
        if orientation == 'axial':
            beta = beta.permute(3,1,0,2)
        elif orientation == 'coronal':
            beta = beta.permute(0,1,3,2).flip(2)
        elif orientation == 'sagittal':
            beta = beta.permute(2,1,3,0).flip(2)
        decoder.decode_single_img(beta, theta, args.out_dir, args.prefix, \
                                  orientation=orientation, volume=True, \
                                  img_hdr=img_hdr, img_affine=img_affine)

    # fusion
    decode_img_dirs = []
    for orientation in orientations:
        decode_img_dirs.append(os.path.join(args.out_dir,
                                            f'{args.prefix}_{orientation}_recon.nii.gz'))
    
    combine_imgs(decode_img_dirs, args.out_dir, args.prefix)

if __name__ == '__main__':
    main()

