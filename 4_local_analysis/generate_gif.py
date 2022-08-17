# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:43:17 2022

@author: 320190618
"""
# prerequisite: having the csv ready 
import numpy as np
from PIL import Image
import glob
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

place_case = 'HamburgUKE-case9'
# get a list of files
flist = glob.glob('C:/Users/320190618/Documents/Original_dicom/' + place_case +'/*')
# for questionable cases 
outputpath = 'C:/Users/320190618/Documents/DSA_dataset/Markedcases/'
csv_dest = 'C:/Users/320190618/Documents/DSA_dataset/acc_folder/' + place_case + '.csv'
df = pd.read_csv(csv_dest, index_col=0)
##load angle as well 
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
# every catogory starts from zero 
label_dict= {}
for i in range(7):
    label_dict[i] = 0
first_enter_dict = {}
for i in range(7):
    first_enter_dict[i] = True    
#%%index to check 
csv_dest = 'C:/Users/320190618/Documents/Dataset-acc/' + 'ds_' + place_case + '.csv'
df_1= pd.read_csv(csv_dest, index_col=0)
ce_index = df_1.loc[df_1['contrast_enhanced']==True].index
#%%
from pydicom import dcmread
# specify index11 12 14 15 16 17 20 21 22 23 24 25 27 28
index = 33
case_num = flist[index][-7:] 
fullpath = 'C:/Users/320190618/Documents/Original_dicom/' + place_case + '/' + case_num
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
import os
marked = True
label = 4
sind = 8
eind = 15
# 0: Arota 1: SMA, 2: Celiac 3: Renal 4: Multiple , 5: SMA + celiac, 6: iliac
# guidewire/catheter only(0)
#endographt delivery sheath(1)
# unsheathed endograft(2)
# with branch stent(3)
# final deployment (4) illac artery 
# big view final (5)

continous = False
# if contrast_enhanced, saving first unaffected frames anf label it 
contrast_enhanced = False
save_index = 2

# duration is the number of milliseconds between frames; this is 2.5 frames per second
if marked == True:
    imgs[0].save(outputpath + "%s.gif"%(place_case + '_' + case_num), save_all=True, append_images=imgs[1:], duration=400, loop=0)
    # df.loc[df['fname'] ==case_num, 'target_folder'] = 'marked'
else:
    
    if continous == True:
        if len(arr.shape) >=3:
            df.loc[df['fname'] ==case_num, 'continous'] = 'True'
            # save to last folder
            curr_folder =  str(label) + '_' + str(label_dict[label])
            df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
            # get the files in the current folder 
            # cur_files = glob.glob('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + last_folder + '/*')
            for i, f in enumerate(imgs_16):
                f.save('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + curr_folder+ '/' + '%04d.png'%(i+start_index))      
            # update start_index 
            start_index = start_index + i+1
        else:
            df.loc[df['fname'] ==case_num, 'continous'] = 'True'
            # save to last folder
            curr_folder =  str(label) + '_' + str(label_dict[label])
            df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
            f = Image.fromarray(arr)
            # get the files in the current folder 
            # cur_files = glob.glob('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + last_folder + '/*')
            f.save('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + curr_folder+ '/' + '%04d.png'%(start_index))      
            # update start_index 
            start_index = start_index +1
            
    elif contrast_enhanced == True:
        
        start_index = 0
        #save to a new folder, start with 2_0
        if first_enter_dict[label] == False:
            # update dict 2_1
            label_dict[label] = label_dict[label] + 1
        curr_folder = str(label) + '_' + str(label_dict[label])
        df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
        df.loc[df['fname'] ==case_num, 'contrast_enhanced'] = 'True'
        # creat curr_folder
        os.mkdir('C:/Users/320190618/Documents/DSA_dataset/' + place_case + '/' + curr_folder)
        for i, f in enumerate(imgs_16[:save_index]):
            f.save('C:/Users/320190618/Documents/DSA_dataset/' + place_case + '/' + curr_folder + '/' + '%04d.png'%(i+start_index))
        start_index = i+1
        first_enter_dict[label] = False
        
    else:
        if len(arr.shape) >=3:
            # start a new folder 
            start_index = 0
            #save to a new folder, start with 2_0
            if first_enter_dict[label] == False:
                # update dict 2_1
                label_dict[label] = label_dict[label] + 1
            curr_folder = str(label) + '_' + str(label_dict[label])
            df.loc[df['fname'] ==case_num, 'target_folder'] = curr_folder
            df.loc[df['fname'] ==case_num, 'sind'] = sind
            df.loc[df['fname'] ==case_num, 'eind'] = eind
            # creat curr_folder
            os.mkdir('C:/Users/320190618/Documents/DSA_dataset/' + place_case + '/' + curr_folder)
            for i, f in enumerate(imgs_16[sind:eind+1]):
                f.save('C:/Users/320190618/Documents/DSA_dataset/' + place_case + '/' + curr_folder + '/' + '%04d.png'%(i+start_index))
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
            os.mkdir('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + curr_folder)
            f = Image.fromarray(arr)
            # get the files in the current folder 
            # cur_files = glob.glob('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + last_folder + '/*')
            f.save('C:/Users/320190618/Documents/FEVAR_dataset/' + place_case + '/' + curr_folder+ '/' + '%04d.png'%(start_index))      
            # update start_index 
            start_index = start_index +1
            
            # single frame can aslo initiate a new folder 
            
# save into a destination folder
# update csv
# if continue as previous, get the last index and continue that index 
# update csv as continues 


# matplotlib.rcParams["figure.dpi"] = 200
# plt.imshow(arr[0], cmap= 'gray')
#%%
output_dir = 'C:/Users/320190618/Documents/DSA_dataset/'
df.to_csv(output_dir + 'dsa_%s.csv'%place_case)