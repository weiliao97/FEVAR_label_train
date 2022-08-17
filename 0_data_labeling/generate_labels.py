# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:55:09 2022

@author: 320190618

Script to label the dataset, 
In total 4 cells, run it cell by cell
"""

import glob
import os
from pydicom import dcmread
import pandas as pd
import numpy as np
from PIL import Image
import glob
import matplotlib 
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# specify which case you are labeling and the directory, number of classes, 
# target directory for labeled images, directory for uncertain images saved as gif
place_case =  'HamburgUKE-case3'
flist = glob.glob('C:/Users/320190618/Documents/Original_dicom/HamburgUKE-case3/*')
num_classes = 5 
target_directory = 'C:/Users/320190618/Documents/code_compile/0_data_labeling/FEVAR_dataset/'
gif_directory = 'C:/Users/320190618/Documents/code_compile/0_data_labeling/Marked_cases/'
acc_directory = 'C:/Users/320190618/Documents/code_compile/0_data_labeling/FEVAR_acc/'
or_directory = 'C:/Users/320190618/Documents/Original_dicom/' 


# a function to display DICOM files frame by frame 
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
        
# read original disom files and useful headers
im_name = [i[-7:] for i in flist]
frames = []
ac_time = []
content_time = []
series_time = []
modality = []
t_angle = []
c_angle = []
for f in flist:
    ds = dcmread(f)
    arr = ds.pixel_array
    modality.append(ds.Modality)
    if 'AcquisitionTime' not in ds:
        ac_time.append('not specificed')
    else: 
        ac_time.append(ds.AcquisitionTime)
    if 'SeriesTime' not in ds:
        series_time.append('not specified')
    else:
        series_time.append(ds.SeriesTime)
    if 'ContentTime' not in ds:
        content_time.append('not specified')
    else:
        content_time.append(ds.ContentTime)
    if ('2003', '2024') not in ds:
        t_angle.append('not specified')
    else:
        t_angle.append(float(ds['2003', '2024'].value))
    if ('2003', '2022') not in ds:
        c_angle.append('not specified')
    else:
        c_angle.append(float(ds['2003', '2022'].value))        
    
    frames.append(arr.shape)

data = {'place_case': [place_case] * len(frames),
        'fname': im_name,  
        'modality': modality, 
        'frames': frames,
        'series_time': series_time,
        'ac_time': ac_time, 
        'content_time': content_time,
        'lateral angle': c_angle,
        'Htf angle': t_angle
        }

df = pd.DataFrame(data)

#%%
# start labeling
# every catogory starts from zero 
label_dict= {}
for i in range(num_classes):
    label_dict[i] = 0
first_enter_dict = {}
for i in range(num_classes):
    first_enter_dict[i] = True    

# specific index to display, after examining the images, decide which label to give 
index = 2
case_num = flist[index][-7:] 
fullpath =  or_directory + place_case + '/' + case_num
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
#%% Assign labels 
label = 1
# guidewire/catheter only(0)
#endographt delivery sheath(1)
# unsheathed endograft(2)
# with branch stent(3)
# final deployment (4) illac artery 

# if marked is True, save into gif for later analysis 
marked = False
# if is_DSA_image is True, saving first unaffected 2 frames (if they are unaffected by the contrast medium)
is_DSA_image = False
save_index = 2

# duration is the number of milliseconds between frames; this is 2.5 frames per second
if marked == True:
    imgs[0].save(gif_directory+ "%s.gif"%(place_case + '_' + case_num), save_all=True, append_images=imgs[1:], duration=400, loop=0)
    df.loc[df['fname'] ==case_num, 'target_folder'] = 'marked'
else:
            
    if is_DSA_image  == True:
        
        start_index = 0
        #save to a new folder, start with 2_0
        if first_enter_dict[label] == False:
            # update dict 2_1
            label_dict[label] = label_dict[label] + 1
        curr_folder = str(label) + '_' + str(label_dict[label])
        df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
        df.loc[df['fname'] ==case_num, 'is_DSA_image'] = 'True'
        # creat curr_folder
        os.mkdir(target_directory + place_case + '/' + curr_folder)
        for i, f in enumerate(imgs_16[:save_index]):
            f.save(target_directory  + place_case + '/' + curr_folder + '/' + '%04d.png'%(i+start_index))
        start_index = i+1
        first_enter_dict[label] = False
        
    else:

        # start a new folder 
        start_index = 0
        #save to a new folder, start with 2_0
        if first_enter_dict[label] == False:
            # update dict 2_1
            label_dict[label] = label_dict[label] + 1
        curr_folder = str(label) + '_' + str(label_dict[label])
        df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
        # creat curr_folder
        os.mkdir(target_directory + place_case + '/' + curr_folder)
        for i, f in enumerate(imgs_16):
            f.save(target_directory  + place_case + '/' + curr_folder + '/' + '%04d.png'%(i+start_index))
        start_index = i+1
        first_enter_dict[label] = False

#%% finally save the csv linking the origunal DICOM to the labeled images 
df.to_csv(acc_directory + '%s.csv'%place_case)