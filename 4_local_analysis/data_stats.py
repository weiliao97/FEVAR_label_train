# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:10:51 2022

@author: 320190618
"""
#%%
duration_dict= {}
for i in range(16):
    label_dict[i] = {}
    for j in range(5):
        dict_0 = {}
        label_dict[i][str(j)] = []  
#%%
# procedure duration 
scan = os.listdir('C:/Users/320190618/Documents/Dataset-acc')
total_csv_file = [i for i in scan if 'ds' in i]
for k, csv_file in enumerate(total_csv_file):
    df= pd.read_csv(os.path.join('C:/Users/320190618/Documents/Dataset-acc', csv_file), index_col=0)
    time_label = df.loc[df.loc[:, 'target_folder'].isnull()==False, ['series_time', 'target_folder']]
    time_label['series_time'] = pd.to_numeric(time_label['series_time'] )
    time_label.sort_values(by=['series_time'], inplace=True)
    labels = time_label['target_folder'].values
    start = labels[0].split('_')[0]
    pre = start
    label_dict[k][start].append(time_label.loc[time_label['target_folder'] == labels[0], 'series_time'].values[0])
    curr = start 
    for i, l in enumerate(labels[1:]):
        if l.split('_')[0] != curr:
            t = time_label.loc[time_label['target_folder'] == l, 'series_time'].values[0]
            label_dict[k][curr].append(t)
            curr = l.split('_')[0]
            label_dict[k][curr].append(time_label.loc[time_label['target_folder'] == l, 'series_time'].values[0])
        if i == len(labels)-2:
            label_dict[k][curr].append(time_label.loc[time_label['target_folder'] == l, 'series_time'].values[0])
#%%
# data and plotting 
duration_dict_sum = {}
for j in range(5):
    duration_dict_sum[j] = []

def diff_times(t1, t2):
    # caveat emptor - assumes t1 & t2 are python times, on the same day and t2 is after t1
    if '.' in t1:
        t1 = t1.split('.')[0]
    if '.' in t2:
        t2 = t2.split('.')[0]

    if len(t1)>=6:
        h1 = int(t1[0:2])
        m1, s1 = int(t1[2:4]), int(t1[4:6]) 
    else:
        h1 = int(t1[0])
        m1, s1 = int(t1[1:3]), int(t1[3:5]) 
    if len(t2)>=6:
        h2 = int(t2[0:2])
        m2, s2 = int(t2[2:4]), int(t2[4:6]) 
    else:
        h2 = int(t2[0])
        m2, s2 = int(t2[1:3]), int(t2[3:5]) 

    t1_secs = s1 + 60 * (m1 + 60*h1)
    t2_secs = s2 + 60 * (m2 + 60*h2)
    return ( t2_secs - t1_secs)/60

for k in range(5):
    for i in label_dict:
        if k==2:
            if len(label_dict[i][str(k)]) >=2: 
                curr = label_dict[i][str(k)]
                curr = [str(i) for i in curr]
                if curr[1] != curr[0]:
                    length =  diff_times(curr[0], curr[1])
                    duration_dict_sum[k].append(length)
        else:
            if len(label_dict[i][str(k)]) >=2: 
                curr = label_dict[i][str(k)]
                curr = [str(i) for i in curr]
                if curr[-1] != curr[0]:
                    length =  diff_times(curr[0], curr[-1])
                    duration_dict_sum[k].append(length)            
        
        # if len(label_dict[i][str(k)]) >=6:
        #     curr = label_dict[i][str(k)]
        #     curr = [str(i) for i in curr]
        #     length = diff_times(curr[4], curr[5]) + diff_times(curr[2], curr[3]) + diff_times(curr[0], curr[1])
        #     duration_dict_sum[k].append(length)
        # elif len(label_dict[i][str(k)]) >=4:
        #     curr = label_dict[i][str(k)]
        #     curr = [str(i) for i in curr]
        #     length = diff_times(curr[2], curr[3]) + diff_times(curr[0], curr[1])
        #     duration_dict_sum[k].append(length)
        # elif len(label_dict[i][str(k)]) >=2: 
        #     curr = label_dict[i][str(k)]
        #     curr = [str(i) for i in curr]
                    
        #     if curr[1] != curr[0]:
        #         length =  diff_times(curr[0], curr[1])
        #         duration_dict_sum[k].append(length)
#%%scipts for plotting 
mean = []
std = []
for k in duration_dict_sum:
    curr = duration_dict_sum[k]
    for c in curr:
        if c <=0:
            curr.remove(c)
    if k == 0:
        for c in curr:
            if c <=1 or c >=110:
                curr.remove(c)
    mean.append(np.mean(curr))
    std.append(np.std(curr))
    
import matplotlib 
import matplotlib.pyplot as plt
plt.style.use ('bmh') 
matplotlib.rcParams["figure.dpi"] =200
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
font = {'weight' : 'bold'} 
plt.rc('font', **font)     

fig, ax = plt.subplots()
ax.bar(range(5), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Procedure duration/min')
ax.set_xticks(range(0, 5))
ax.set_xticklabels(['Navigation','Sheath delivery','Unsheathed','Cannulation', 'Final deployment'], \
                   rotation=30)

#%%time_label
#walk through folders 
import os

filenames= os.listdir (r"C:\Users\320190618\Documents\DSA_dataset") # get all files' and folders' names in the current directory

result = []
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(r"C:\Users\320190618\Documents\DSA_dataset", filename)): # check whether the current object is a folder or not
        result.append(filename)

result.sort()
print(result)

#%%
for i in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    fname = result[i]
    csv_dest ='C:/Users/320190618/Documents/DSA_acc-4cat/' + 'dsa_' + fname + '.csv'
    df= pd.read_csv(csv_dest, index_col=0)
    cat_names = glob.glob('C:/Users/320190618/Documents/DSA_dataset/' + fname + '/*')
    all_cat_name = [c.split('\\')[-1].split('_')[0] for c in cat_names]
    
    cat125 = [j for j in all_cat_name if j in ['1', '2', '5']]
    cat125ind = [m for m in range(len(all_cat_name)) if all_cat_name[m] in ['1', '2', '5']]
    cat125_name = [cat_names[p] for p in cat125ind]
    onlycat_name = [c.split('\\')[-1] for c in cat_names]
    if len(cat125) > 0:
        for k, m in enumerate(cat125):
            print(cat125_name[k])
            to_t = str(0) + '_' + str(k)
            print(to_t)
            os.rename(cat125_name[k], os.path.join('C:/Users/320190618/Documents/DSA_dataset/', \
                                                   fname, to_t))
            df.loc[df['target_folder'] == onlycat_name [k] , 'target_test'] = to_t
    
    cat3 = [j for j in all_cat_name if j == '3']
    cat3ind = [m for m in range(len(all_cat_name)) if all_cat_name[m] == '3']
    cat3_name = [cat_names[p] for p in cat3ind]
    onlycat_name = [c.split('\\')[-1] for c in cat3_name]
    if len(cat3) > 0:
        for k, m in enumerate(cat3):
            print(cat3_name[k])
            to_t = str(1) + '_' + str(k)
            print(to_t)
            os.rename(cat3_name[k], os.path.join('C:/Users/320190618/Documents/DSA_dataset/', \
                                                   fname, to_t))
            df.loc[df['target_folder'] == onlycat_name[k] , 'target_test'] = to_t
    
    cat4 = [j for j in all_cat_name if j == '4']
    cat4ind = [m for m in range(len(all_cat_name)) if all_cat_name[m] == '4']
    cat4_name = [cat_names[p] for p in cat4ind]
    onlycat_name = [c.split('\\')[-1] for c in cat4_name]
    if len(cat4) > 0:
        for k, m in enumerate(cat4):
            print(cat4_name[k])
            to_t = str(3) + '_' + str(k)
            print(to_t)
            os.rename(cat4_name[k], os.path.join('C:/Users/320190618/Documents/DSA_dataset/', \
                                                   fname, to_t))
            df.loc[df['target_folder'] == onlycat_name[k] , 'target_test'] = to_t

    cat6 = [j for j in all_cat_name if j == '6']
    cat6ind = [m for m in range(len(all_cat_name)) if all_cat_name[m] == '6']
    cat6_name = [cat_names[p] for p in cat6ind]
    onlycat_name = [c.split('\\')[-1] for c in cat6_name]
    if len(cat6) > 0:
        for k, m in enumerate(cat6):
            print(cat6_name[k])
            to_t = str(2) + '_' + str(k)
            print(to_t)
            os.rename(cat6_name[k], os.path.join('C:/Users/320190618/Documents/DSA_dataset/', \
                                                   fname, to_t))
            df.loc[df['target_folder'] == onlycat_name[k] , 'target_test'] = to_t
    
    output_dir = 'C:/Users/320190618/Documents/DSA_dataset/acc_folder/'
    df.to_csv(output_dir + 'dsa_%s.csv'%fname)
    
            
# result.remove('Markedcases')
# result.remove('Removed')
#%%


#%%
label_dict= {}
for i in range(4):
    s = str(i)
    label_dict[s] = 0
#%%
frame_dict= {}
for i in range(4):
    s = str(i)
    frame_dict[s] = 0
    
frames = []
#%%
r_dict = []
for r in result:
    frame_dict= {}
    for i in range(4):
        s = str(i)
        frame_dict[s] = 0
    r_dict.append(frame_dict)
    
#%%
import glob 
for i, r in enumerate(result):
    cat_names = glob.glob('C:/Users/320190618/Documents/DSA_dataset/' + r + '/*')
    for c in cat_names:
        c_0 = c.split('\\')[-1].split('_')[0]
        frames_num = glob.glob( c + '/*')
        # frames_clip = get_clip_num(len(frames_num))
        r_dict[i][c_0] += 1
        label_dict[c_0] += 1
        #number of frames:
        frame_names = glob.glob(c + '/*')
        frame_dict[c_0] += len(frame_names)
        if len(frame_names) > 10:
            print(r)
            print(c)
        frames.append(len(frame_names))
        #count frames in each subfolder and 
#%% scipts for plotting 
import matplotlib 

import matplotlib.pyplot as plt
plt.style.use ('bmh') 
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
font = {'weight' : 'bold'} 
plt.rc('font', **font)     

names = list(label_dict.keys())
values = list(label_dict.values())
plt.bar(range(len(label_dict)), values, tick_label=names)
plt.xticks(ticks = [0, 1, 2, 3], labels = ['SMA/Celiac', 'Renal', 'Iliac', 'All'], 
           rotation=15)
# 'GW_only, Sheath', 'Unsheathed, 'Branch\nstent', 'Final\ndeployment'
# def autolabel(rects):
#     for ii,rect in enumerate(rects):
#         height = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% (mean_values[ii]),
#             ha='center', va='bottom')
# autolabel(rects)

# plt.xlabel('Category', fontweight='bold')
plt.ylabel('# of Clips', fontweight='bold')
plt.show()  
#%%    
names = list(frame_dict.keys())
values = list(frame_dict.values())
plt.bar(range(len(label_dict)), values, tick_label=names)
# plt.xlabel('Category', fontweight='bold')
plt.ylabel('# of Frames', fontweight='bold')
plt.xticks(ticks = [0, 1, 2, 3], labels = ['SMA/Celiac', 'Renal', 'Iliac', 'All'], 
           rotation=15)
plt.show()   
#%% frame length stats 
matplotlib.rcParams["figure.dpi"] = 200
plt.hist(frames, bins=[2,  10, 50, 100, 200, 360])
plt.ylabel('# of videos', fontweight='bold')
plt.xlabel('# of frames/video', fontweight='bold')
#%% how to split train and validation 
# 5 fold splits by index 
# statics by folder 
def split(x, n):
    seg = []
 
    # If x % n == 0 then the minimum
    # difference is 0 and all
    # numbers are x / n
    if (x % n == 0):
        for i in range(n):
            seg.append(x//n)
    else:
        # upto n-(x % n) the values
        # will be x / n
        # after that the values
        # will be x / n + 1
        zp = n - (x % n)
        pp = x//n
        for i in range(n):
            if(i>= zp):
                seg.append(pp+1)
            else:
                seg.append(pp)
    return seg 

def get_clip_num(l):
    if l > 200: 
        l_skip = int((l+1)/4)
        l_seg = int(l_skip/10) + 1 
        seg = split(l_skip, l_seg) # [9, 9, 9, 10]
    elif l>100:
        l_skip = int((l+1)/2)
        l_seg = int(l_skip/10) + 1 
        seg = split(l_skip, l_seg) # [9, 9, 9, 10]
    elif l > 10: # split (11, 2) = [5, 6]
        l_seg = int(l/10) + 1 
        seg = split(l, l_seg) 
    else:
        seg = [0]
    return len(seg)


import numpy as np 
folder_num = np.zeros((17, 5))
for i, r in enumerate(result):
    cat_names = glob.glob('C:/Users/320190618/Documents/FEVAR_dataset/' + r + '/*')
    for j, c in enumerate(cat_names):
        c_0 = c.split('\\')[-1].split('_')[0]
        # c could be C:/Users/320190618/Documents/FEVAR_dataset/Utrecht-UMCU-case4\4_0
        # find frame numbers under c 
        frames_num = glob.glob( c + '/*')
        frames_clip = get_clip_num(len(frames_num))
        folder_num[i, int(c_0)]  += frames_clip
        # label_dict[c_0] += 1
        # #number of frames:
        # frame_names = glob.glob(c + '/*')
        # frame_dict[c_0] += len(frame_names)
        # frames.append(len(frame_names))
    