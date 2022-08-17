# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:55:09 2022

@author: 320190618
"""

# add annotation 
import glob
place_case =  'HamburgUKE-case3'
flist = glob.glob('C:/Users/320190618/Documents/Original_dicom/HamburgUKE-case3/*')
im_name = [i[-7:] for i in flist]
from pydicom import dcmread
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

#%%
import pandas as pd

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

info = pd.DataFrame(data)
#%%
output_dir = 'C:/Users/320190618/Documents/FEVAR_dataset/'
info.to_csv(output_dir + 'angle_%s.csv'%place_case)