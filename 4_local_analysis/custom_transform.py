# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:08:09 2022

@author: 320190618
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Sampler
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# import torchvision.transforms as transforms
# from tqdm import tqdm
import cv2 
from collections import defaultdict
import random 
import torchvision.transforms as transforms
import matplotlib
matplotlib.rcParams["figure.dpi"] = 200
def pad_img(img):
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1] # 968 - 750 = 218
        ld = int(diff/2)
        rd = diff - ld
        padded_img = cv2.copyMakeBorder(img, 0, 0, ld, rd, borderType= cv2.BORDER_CONSTANT)
    else:
        diff = img.shape[1] - img.shape[0]
        td = int(diff/2)
        bd = diff -td
        padded_img = cv2.copyMakeBorder(img, td, bd, 0, 0, cv2.BORDER_CONSTANT)
    return padded_img


path = 'C:/Users/320190618/Documents/FEVAR_dataset'
name = 'HamburgUKE-case6/3_0'
frames = os.listdir(os.path.join(path, name))
pix = np.asarray([[ 739, 543], [1333, 1497]])
img_size = 512
train_transform = transforms.Compose([transforms.Resize([img_size, img_size]),
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
# aug_transform =  transforms.Compose([transforms.RandomPerspective(distortion_scale=0.6, p=0.3), 
#                                 ])
aug_transform =  transforms.Compose([transforms.RandomApply(torch.nn.ModuleList([
     transforms.RandomRotation((-30, 30)), ]), p=0.5), 
     transforms.RandomPerspective(distortion_scale=0.2),
     transforms.RandomAdjustSharpness(sharpness_factor=2),
     transforms.RandomAutocontrast(),
     transforms.RandomHorizontalFlip()
    ])
# scripted_transforms = torch.jit.script(aug_transform)
X_padded = torch.zeros((len(frames), 1, img_size, img_size))
for i, f in enumerate(frames[:-1]):
    frame = Image.open(os.path.join(path, name, '{:04d}.png'.format(i)))
    # pix = np.load(r'C:\Users\320190618\Documents\FEVAR_dataset\HamburgUKE-case1\3_6\pix.npy')
    # while it's still PIL, Convert to (H, W, C)  
    frame_8 = (np.array(frame)*255/65535).astype('uint8')
    frame_crop =  frame_8[pix[0, 1] : pix[1, 1], pix[0, 0] : pix[1, 0]]
    # check size:
    if frame_crop.shape[0] != frame_crop.shape[1]:
        frame_pad = pad_img(frame_crop)
        # frame_3d = np.expand_dims(frame_pad, axis = -1)
    # else:
    #     frame_3d = np.expand_dims(frame_crop, axis = -1)
    # frame_rgb = np.repeat(frame_3d, 3, axis= -1)
    # rgb_f = transforms.Grayscale(num_output_channels=3)(frame)
    # rgb_8 = (np.array(rgb_f)/256).astype('uint8')
    rgb_pil = Image.fromarray (frame_pad, mode = 'L')
    frame_trans = train_transform (rgb_pil)
    X_padded[i, :] = frame_trans
X_aug = aug_transform(X_padded)
    
plt.figure()
plt.imshow(frame_trans[0], cmap='Greys')
plt.figure()
plt.imshow(frame_crop, cmap='Greys')
plt.figure()
plt.imshow(frame_8, cmap='Greys')
plt.figure()
plt.imshow(X_aug[0, 0], cmap='Greys')
plt.figure()
plt.imshow(X_aug[1, 0], cmap='Greys')

#%%
formatted = (frame_trans[0][180:260, 220:350].numpy() * 255/np.max(frame_trans[0].numpy())).astype('uint8')
image = Image.fromarray(formatted)