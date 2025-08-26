#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.module1 = nn.Sequential()
        self.module2 = nn.Sequential()
        self.module3 = nn.Sequential()
        self.module4 = nn.Sequential()
    
        for x in range(4):
            self.module1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.module2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.module3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.module4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.module1(X)
        h_relu1_2 = h
        h = self.module2(h)
        h_relu2_2 = h
        h = self.module3(h)
        h_relu3_3 = h
        h = self.module4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
        
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, num_lvs=4, base_ch=16, final_act='noact'):
        super(Unet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_lvs = num_lvs
        self.base_ch = base_ch
        self.final_act = final_act

        self.in_conv = nn.Conv2d(in_ch, self.base_ch, 3, 1, 1)
        
        for lv in range(self.num_lvs):
            channel = self.base_ch * (2 ** lv)
            self.add_module(f'downconv_{lv}', ConvBlock2d(channel, channel*2, channel*2))
            self.add_module(f'maxpool_{lv}', nn.MaxPool2d(kernel_size=2, stride=2))
            self.add_module(f'upsample_{lv}', Upsample(channel*4))
            self.add_module(f'upconv_{lv}', ConvBlock2d(channel*4, channel*2, channel*2))

        bttm_ch = self.base_ch * (2 ** self.num_lvs)
        self.bttm_conv = ConvBlock2d(bttm_ch, bttm_ch*2, bttm_ch*2)

        self.out_conv = nn.Conv2d(self.base_ch*2, self.out_ch, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, in_tensor):
        concat_out = {}
        x = self.in_conv(in_tensor)
        for lv in range(self.num_lvs):
            concat_out[lv] = getattr(self, f'downconv_{lv}')(x)
            x = getattr(self, f'maxpool_{lv}')(concat_out[lv])
        x = self.bttm_conv(x)
        for lv in range(self.num_lvs-1, -1, -1):
            x = getattr(self, f'upsample_{lv}')(x, concat_out[lv])
            x = getattr(self, f'upconv_{lv}')(x)
        x = self.out_conv(x)
        x = self.leakyrelu(x)
        return x

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU())

    def forward(self, in_tensor):
        return self.conv(in_tensor)

class Upsample(nn.Module):
    def __init__(self, in_ch):
        super(Upsample, self).__init__()
        self.out_ch = int(in_ch / 2)
        self.conv = nn.Conv2d(in_ch, self.out_ch, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(self.out_ch, affine=True)
        self.act = nn.LeakyReLU()

    def forward(self, in_tensor, ori):
        upsmp = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=True)
        upsmp = self.act(self.norm(self.conv(upsmp)))
        output = torch.cat((ori, upsmp), dim=1)
        return output

class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ThetaEncoder, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(), # 288*288 --> 144*144
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(), # 144*144 --> 72*72
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(), # 72*4872--> 36*36
            nn.Conv2d(128, self.out_ch, 4, 2, 1),
            nn.InstanceNorm2d(self.out_ch, affine=True),
            nn.LeakyReLU()) # 36*36--> 18*18

    def forward(self, in_tensor):
        return self.conv(in_tensor)

class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, 16, 4, 2, 1)), 
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(), #288,288 --> 144,144
            nn.utils.spectral_norm(nn.Conv2d(16, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(), # 144,144 --> 72,72
            nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(), # 72,72 --> 36,36
            nn.utils.spectral_norm(nn.Conv2d(64, 16, 1, 1, 0)),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 1, 1, 1, 0)))

    def forward(self, in_tensor):
        return self.seq(in_tensor)

class DomainAdaptorBeta(nn.Module):
    def __init__(self, in_ch, out_ch, final_act=True):
        super(DomainAdaptorBeta, self).__init__()
        self.final_act = final_act
        self.da = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1))
        self.tanh = nn.Tanh()
        
    def forward(self, in_tensor):
        if self.final_act:
            return self.tanh(self.da(in_tensor))
        else:
            return self.da(in_tensor)

class DomainAdaptorTheta(nn.Module):
    def __init__(self, out_ch):
        super(DomainAdaptorTheta, self).__init__()
        self.out_ch = out_ch
        self.mean_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.out_ch, 18, 18, 0))
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.out_ch, 18, 18, 0))

    def forward(self, in_tensor, device):
        mu = self.mean_conv(in_tensor)
        logvar = self.logvar_conv(in_tensor)
        theta = self.sample(mu, logvar, device)
        #print(theta.shape)
        return theta, mu, logvar

    def sample(self, mu, logvar, device):
        theta = torch.randn(mu.size()).to(device) * torch.sqrt(torch.exp(logvar)) + mu
        return theta
        
class FusionNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FusionNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.net = nn.Sequential(
            nn.Conv3d(self.in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.out_conv = nn.Sequential(
            nn.Conv3d(16+self.in_ch, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, self.out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, in_tensor):
        return self.out_conv(torch.cat([in_tensor,self.net(in_tensor)], dim=1))
