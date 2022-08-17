# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:22:00 2022

@author: 320190618
"""
#%%
from moviepy.video.io import VideoFileClip
from moviepy.video.fx.all import crop
# use pydicom for reading 
vfile = r'C:\Users\320190618\OneDrive - Philips\Video_clips\AT_26102021_085258.568.mp4'
clip = VideoFileClip.VideoFileClip(vfile).subclip(50,60)
cropped = crop(clip, x1=50, y1=60, x2=460, y2=275)
