# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:56:39 2022

@author: 320190618
"""
import numpy as np
from PIL import Image
import glob
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
#%%
place = 'Utrecht-UMCU-case5'
img = 'IM_0011'
fullpath = 'C:/Users/320190618/Documents/Original_dicom/' + place + '/' + img
ds = dcmread(fullpath)
arr = ds.pixel_array
imgs = [Image.fromarray(img/255) for img in arr]
imgs_16 = [Image.fromarray(img) for img in arr]
# plotting 
fig, ax = plt.subplots(1,1, figsize=(10, 10))
if len(arr.shape) >=3:
    tracker = IndexTracker(ax, arr)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
else:
    plt.imshow(arr)

#%%
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()
#%%
label = 5
label_ind = 0
sind = 6
eind = 7
start_index = 0
curr_folder = str(label) + '_' + str(label_ind)
# creat curr_folder
os.mkdir('C:/Users/320190618/Documents/DSA_dataset/' + place+ '/' + curr_folder)
for i, f in enumerate(imgs_16[sind:eind+1]):
    f.save('C:/Users/320190618/Documents/DSA_dataset/' + place + '/' + curr_folder + '/' + '%04d.png'%(i+start_index))