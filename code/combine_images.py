#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import numpy as np
from modules.fusion import FusionNetwork
import nibabel as nib
from torchvision.transforms import ToTensor

def load_images(img_dirs):
    imgs = []
    for img_dir in img_dirs:
        img_file = nib.load(img_dir)
        img = np.array(img_file.get_fdata().astype(np.float32))
        img = ToTensor()(img)
        imgs.append(img.float().permute(2,1,0).permute(2,0,1).unsqueeze(0).unsqueeze(0))
    return imgs, img_file.affine, img_file.header
        
def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='combine images')
    parser.add_argument('--in-imgs', type=str, nargs='+', required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--pretrained-model', type=str, required=True)
    parser.add_argument('--norm', default=0.25, type=float)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(args)

    trainer = FusionNetwork(pretrained_model = args.pretrained_model,
                            gpu = args.gpu)
    in_imgs, img_affine, img_hdr = load_images(img_dirs = args.in_imgs)
        
    trainer.test(imgs=in_imgs, out_dir=args.out_dir, prefix=args.prefix,
                 img_affine=img_affine, img_hdr=img_hdr, norm=args.norm)

if __name__ == '__main__':
    main()
