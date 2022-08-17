# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:34:35 2022

@author: 320190618
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_cam_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
    return result

#%%
y = np.load('C:/Users/320190618/Documents/Video_annot/dsa_val1619/y.npy')
y_pred = np.load('C:/Users/320190618/Documents/Video_annot/dsa_val1619/y_pred.npy')
cam_t = np.load('C:/Users/320190618/Documents/Video_annot/dsa_val1619/cam_t.npy')
image_t = np.load('C:/Users/320190618/Documents/Video_annot/dsa_val1619/image_t.npy')
l_map = [tensor([10,  7, 10,  5]), tensor([10,  3,  7,  4]), tensor([10, 10,  4, 10,  6]), tensor([5, 8, 4, 7]), tensor([4, 3, 5, 4]), tensor([ 2,  5,  7, 10,  5]), tensor([7, 9, 4, 4])]
#%%6
# 29, 66
import matplotlib.pyplot as plt
index = 64
img = np.repeat(np.expand_dims(image_t[index, 0], -1), 3, -1)
visualization = show_cam_on_image(img, cam_t[index], use_rgb=True)
plt.imshow(visualization)
plt.axis('off')
#%%
plt.imshow(image_t[29, 0], cmap='gray')
plt.axis('off')