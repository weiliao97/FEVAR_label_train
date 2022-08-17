# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% setup 
filepath = r"Z:\BIDMC\20211217_Pat5010 PMEG\Viewing Software\DICOM data\DICOM\IM_0500"
from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np 
import matplotlib 
%matplotlib inline
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 200
#%%
# read files 
ds = dcmread(filepath)
arr = ds.pixel_array
plt.imshow(arr, cmap='gray')

