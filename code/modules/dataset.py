import os
from glob import glob
import nibabel as nib
import torch
import numpy as np
import fnmatch
import random
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, Resize, ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data.dataset import Dataset

default_transform = Compose([ToPILImage(), Pad(60), CenterCrop((288,288))])

class MultiOrientationImages(Dataset):
    """
    To train 3D fusion network
    INPUT
    * dataset_dir: format: '[dataset_dir]/train' and  '[dataset_dir]/valid'
    * data_name: 'T1' or 'T2'
    * Note: file-prefix should be '*_ori.nii.gz' for original images and '*[axial/coronal/sagittal]_recon.nii.gz' for self-recon images 
    """
    def __init__(self, dataset_dir, data_name, mode='train'):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.data_name = data_name
        self.ori_imgs = self._get_files()

    def _get_files(self):
        ori_imgs = os.path.join(self.dataset_dir, self.mode, f'*{self.data_name}_ori.nii.gz')
        ori_imgs = sorted(glob(ori_imgs))
        return ori_imgs

    def __len__(self):
        return len(self.ori_imgs)

    def __getitem__(self, idx: int):
        imgs = []
        orientations = ['axial', 'coronal', 'sagittal']
        ori_img = self.ori_imgs[idx]

        ori_prefix = os.path.basename(ori_img).replace('_ori.nii.gz', '')
        search_dir = os.path.dirname(ori_img)

        for orientation in orientations:
            pattern = os.path.join(search_dir, f'{ori_prefix}_{orientation}_recon.nii.gz')
            img_list = sorted(glob(pattern))
            if not img_list:
                raise FileNotFoundError(f'Cannot find {orientation} recon for {ori_prefix} in {search_dir}')
            img_path = img_list[0]
            img = np.array(nib.load(img_path).get_fdata().astype(np.float32))
            img = ToTensor()(img)
            imgs.append(img.float().permute(2,1,0).permute(2,0,1).unsqueeze(0))

        ori_img_file = nib.load(ori_img)
        ori_img = np.array(ori_img_file.get_fdata().astype(np.float32))
        ori_img = ToTensor()(ori_img)
        return imgs, ori_img.float().permute(2,1,0).permute(2,0,1).unsqueeze(0)


class PairedMRI(Dataset):
    """
    To train harmonization network
    """
    def __init__(self, dataset_dirs, data_names, orientations, mode='train'):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.data_names = data_names
        self.orientations = orientations
        self.imgs, self.dataset_ids = self._get_files()

    def _get_files(self):
        imgs = []
        dataset_ids = []
        for data_name in self.data_names:
            img_list = []
            dataset_id_list = []
            for dataset_id, dataset_dir in enumerate(self.dataset_dirs):
                for orientation in self.orientations:
                    full_path = os.path.join(dataset_dir, self.mode, f'*{data_name}*{orientation.upper()}*.nii.gz')
                    for img_path in sorted(glob(full_path)):
                        img_list.append(img_path)
                        dataset_id_list.append(dataset_id if data_name=='T1' else -1.0*dataset_id)
            imgs.append(img_list)
            dataset_ids.append(dataset_id_list)
        return tuple(imgs), dataset_ids

    def __len__(self):
        return len(self.imgs[0])

    def __getitem__(self, idx:int):
        imgs = []
        dataset_ids = []
        other_imgs = []
        for modality_id in range(len(self.imgs)):
            img_path = self.imgs[modality_id][idx]
            img = nib.load(img_path).get_fdata().astype(np.float32).transpose([1,0])
            msk = img == 0
            img = default_transform(img)
            img = ToTensor()(np.array(img))
            dataset_id = self.dataset_ids[modality_id][idx]

            if 'AXIAL' in img_path:
                str_id = img_path.find('_AXIAL')
            elif 'CORONAL' in img_path:
                str_id = img_path.find('_CORONAL')
            else:
                str_id = img_path.find('_SAGITTAL')

            pattern = img_path[:str_id]+'*AXIAL_SLICE11*.nii.gz'
            pattern2 = img_path[:str_id]+'*AXIAL_SLICE12*.nii.gz'
            pattern3 = img_path[:str_id]+'*AXIAL_SLICE13*.nii.gz'
            pattern0 = img_path[:str_id]+'*AXIAL_SLICE10*.nii.gz'

            other_img_path = random.choice(
                fnmatch.filter(self.imgs[modality_id], pattern) +
                fnmatch.filter(self.imgs[modality_id], pattern2) +
                fnmatch.filter(self.imgs[modality_id], pattern3) +
                fnmatch.filter(self.imgs[modality_id], pattern0)
            )
            other_img = nib.load(other_img_path).get_fdata().astype(np.float32).transpose([1,0])
            other_img = default_transform(other_img)
            other_img = ToTensor()(np.array(other_img))

            imgs.append(img)
            dataset_ids.append(dataset_id)
            other_imgs.append(other_img)

        # make sure both T1 and T2 have the same FOV
        img0 = imgs[0]
        img1 = imgs[1]
        msk0 = img0.ge(1e-3)
        msk1 = img1.ge(1e-3)
        msk = msk0 & msk1
        imgs[0][~msk] = 0
        imgs[1][~msk] = 0

        return tuple(imgs), dataset_ids, tuple(other_imgs)
