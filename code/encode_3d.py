#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from modules.model import CALAMITI

def obtain_single_img(img_dir, avg_theta, norm):
    import torch
    import nibabel as nib
    from PIL import Image
    import numpy as np
    from torchvision.transforms import ToTensor, CenterCrop, Compose, ToPILImage

    img_file = nib.load(img_dir)
    img_vol = np.array(img_file.get_fdata().astype(np.float32))
    img_vol = img_vol / norm * 0.25
    n_row, n_col, n_slc = img_vol.shape
    # obtain image volume for calcualting beta
    zero_img = np.zeros((288,288,288)).astype(np.float32)
    zero_img[144-n_row//2:144+n_row//2+n_row%2, 144-n_col//2:144+n_col//2+n_col%2, 144-n_slc//2:144+n_slc//2+n_slc%2] = img_vol
    img_vol = zero_img
    if avg_theta:
        img_slcs = []
        for slc in range(134,153):
            img_slc = ToTensor()(img_vol[:,:,slc]).unsqueeze(0).permute(0,1,3,2)
            img_slc = img_slc.repeat(img_vol.shape[0],1,1,1)
            img_slcs.append(img_slc)
    else:
        img_slcs = ToTensor()(img_vol[:,:,140]).unsqueeze(0).permute(0,1,3,2)
        img_slcs = img_slcs.repeat(img_vol.shape[0],1,1,1)
    return ToTensor()(img_vol), img_slcs, img_file.header, img_file.affine

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='DisentangledVAE')
    parser.add_argument('--in-img', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='sub1')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--avg-theta', default=False, action='store_true')
    parser.add_argument('--norm', default=1000, type=float)
    args = parser.parse_args(args)

    # initialize model
    encoder = CALAMITI(beta_dim = args.beta_dim,
                       theta_dim = args.theta_dim,
                       train_sample = 'st_gumbel_softmax',
                       valid_sample = 'argmax',
                       pretrained_model = args.pretrained_model,
                       gpu = args.gpu)

    img_vol, img_slc, img_hdr, img_affine = obtain_single_img(img_dir=args.in_img, avg_theta=args.avg_theta, norm=args.norm)
    img_vol = img_vol.float().permute(2,1,0)
    
    # axial
    encoder.encode_single_img(img_vol.permute(2,0,1), img_slc, args.out_dir, args.prefix,
                              orientation='axial', volume=True,
                              img_hdr=img_hdr, img_affine=img_affine)
    # coronal
    encoder.encode_single_img(img_vol.permute(0,2,1).flip(1), img_slc, args.out_dir, args.prefix,
                              orientation='coronal', volume=True,
                              img_hdr=img_hdr, img_affine=img_affine)
    # sagittal
    encoder.encode_single_img(img_vol.permute(1,2,0).flip(1), img_slc, args.out_dir, args.prefix,
                              orientation='sagittal', volume=True,
                              img_hdr=img_hdr, img_affine=img_affine)
if __name__ == '__main__':
    main()
