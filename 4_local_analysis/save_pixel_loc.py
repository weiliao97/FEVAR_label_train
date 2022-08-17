# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:51:56 2022

@author: 320190618
"""

import urllib
import cv2
import os
import glob 
import numpy as np
#the [x, y] for each right-click event will be stored here
right_clicks = list()

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    #right-click event value is 2
    if event == 2:
        global right_clicks

        #store the coordinates of the right-click event
        right_clicks.append([x, y])
        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(right_clicks)
        
        
path = 'C:/Users/320190618/Documents/FEVAR_dataset'
filenames= os.listdir(path) # get all files' and folders' names in the current directory
all_folders = []
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(path, filename)): # check whether the current object is a folder or not
       all_folders.append(filename)
all_folders.sort()
all_cat = glob.glob(path + '/*')
#%%
#get roi based on [1, 4, 15]
dev_folders = [all_folders[i] for i in [1, 4, 15]]
dev_roi = df_1.loc[df_1['Folder'].isin(dev_folders), 'ROI'].tolist()
#%%
cat_name_list_simple = [] 
folder_list = []
for folder in all_folders:
    print(folder)
    all_cat = os.listdir(os.path.join(path, folder))
    for i, cat_name in enumerate(all_cat):
        folder_list.append(folder)
        cat_name_list_simple.append(cat_name)
#%%
cat_name_list = [] 
total_clicks = []
for folder in all_folders[13:]:
    print(folder)
    all_cat = glob.glob(path + '/' + folder + '/*')
    for i, cat_name in enumerate(all_cat):
        cat_name_list.append(cat_name)
        right_clicks = []
        img = cv2.imread(os.path.join(cat_name, '0000.png'),0)
        scale_width = 900/ img.shape[1]
        scale_height = 900 / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', window_width, window_height)
        
        #set mouse callback function for window
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == ord('s'):
            print(right_clicks)
            total_clicks.append(np.asarray(right_clicks))
            # np.save(os.path.join(cat_name, 'pix.npy'), np.asarray(right_clicks))
        if k == ord('c'):
            cv2.destroyAllWindows()
#%%
df_1 = pd.DataFrame(list(zip(folder_list, cat_name_list_simple, total_clicks)), columns =['Folder', 'Cat', 'ROI'])
df_1.to_pickle('C:/Users/320190618/Documents/FEVAR_dataset/crop.pkl')
#%%
with open('C:/Users/320190618/Documents/Video_annot/Video_classify/dev.pkl', "rb") as fh:
  data = pickle.load(fh)
#%%
min_list = []
for i in total_clicks:
    min_list.append((i[1, 0] - i[0, 0]) + (i[1, 1] - i[0, 1]))
print(max(min_list))
# edge detection 
# Convert to graycsale
#%%
img_blur = cv2.GaussianBlur(img, (3,3), 0)
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
scale_width = 1024 / img.shape[1]
scale_height = 1024 / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
