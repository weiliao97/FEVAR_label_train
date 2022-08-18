# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:55:22 2022

@author: 320190618
"""
# import libraries 
# import libraries 
import argparse
import os
import numpy as np 
import pandas as pd 
import torch
import torchvision.transforms as transforms
from torch.utils import data

from prepare_data import *
from sklearn.metrics import accuracy_score

def lis(arr):
    '''
    Parameters
    ----------
    arr : TYPE: list to be examined
        DESCRIPTION. e.g. [0, 1, 2, 3, 4, 3, 5, 6, 5, 7]

    Returns
    -------
    maximum : TYPE: int
        DESCRIPTION. maximum length of the increasing array 
    ind_t : TYPE: list of list
        DESCRIPTION. indices while searching
    lis : TYPE list 
        DESCRIPTION. length of the least increasing array (LIS) ending at that index
    final_ind : TYPE: list
        DESCRIPTION. the indices satisfying LIS

    '''
    n = len(arr)
    # Declare the list (array) for LIS and
    # initialize LIS values for all indexes
    lis = [1]*n
 
    # Compute optimized LIS values in bottom up manner
    ind_t = []
    for i in range(1, n):
        ind = []
        for j in range(0, i):
            if arr[i] >= arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j]+1
                ind.append(j)
        ind_t.append(ind)
 
    # Initialize maximum to 0 to get
    # the maximum of all LIS
    maximum = 0
 
    # Pick maximum of all LIS values
    for i in range(n):
        maximum = max(maximum, lis[i])
    
    # find index 
    ii = np.argmax(lis)
    final_ind = [ii]
    while ii >= 1:
        ii = ind_t[ii-1][-1]
        final_ind.append(ii)     
    return maximum, ind_t, lis, final_ind

def skip_video(train_length):
    '''
    Parameters
    ----------
    train_length : TYPE: int 
        DESCRIPTION. length of the image series

    Returns
    -------
    skip : TYPE: int
        DESCRIPTION. length of the images series after skipping

    '''
    skip = []
    for i in train_length:
        if i >=100 and i < 200:
            skip.append(int((i+1)/2))
        elif i>=200: 
            skip.append(int((i+1)/4))
        else: 
            skip.append(i)
    return skip

def generate_buckets(bs, train_hist):
    '''
    Parameters
    ----------
    bs : TYPE, int
        DESCRIPTION. batch size 
    train_hist : TYPE list 
        DESCRIPTION. histogram of the train or validation set 

    Returns
    -------
    buckets : TYPE list 
        DESCRIPTION. list defining afew intervals such as [0, 5, 15, 50, 100] to group image series

    '''
    buckets = []
    sum = 0
    s = 0
    for i in range(0, 100): 
        # train_hist[0] is len [0, 1), train_hist[354] is  [354, 355]
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i)
            sum = 0 
    # residue is 58 < 128, remove index 205, attach 217, largest is 216
    if sum < bs:
        buckets.pop(-1)    
    buckets.append(101)
    return buckets


def get_name_length(datapath, curr_folder):
        
    '''
    A function to load FEVAR validation cases information 
    Parameters
    ----------
    datapath : TYPE, str
        DESCRIPTION. datapath of the folders to be analyzed, e.g. 'C:/Users/320190618/Documents/GUI_analysis' 
    curr_folder : TYPE, str 
        DESCRIPTION. specific folder to be analyzed e.g. 

    Returns
    -------
    all_names : TYPE, list
        DESCRIPTION. a list of folder names 
        e.g. [''C:/Users/320190618/Documents/GUI_analysis\\BIDMC-case2\\0_0', ...]
    all_length : TYPE, list 
        DESCRIPTION. a list of DICOM series lengths, [12, 10, 14, ...]
    all_y : TYPE, list 
        DESCRIPTION. a list of ground truth labels [0, 0, 0, 2, 2, 2, 2...]
    all_time : TYPE list 
        DESCRIPTION. a list of time points found in the DICOM header 

    '''
    all_names = []
    all_length = []    # each video length
    all_y = []
    all_time = []
    # find series time 
    curr_case =  curr_folder.split('\\')[-1]
    df = pd.read_csv(os.path.join(datapath, 'ds_' + curr_case + '.csv'), index_col=0)
    series_time = df.loc[df['target_folder'].isna() == False, ['series_time', 'target_folder']]
    series_time = series_time.drop_duplicates(subset=['target_folder'], keep='first')
    # f names 
    frame_name  = os.listdir(os.path.join(datapath, curr_folder))
    for f in frame_name: # (4_0, 4_1, 4_2)
        all_names.append(os.path.join(datapath, curr_folder, f))
        sub_folder = os.listdir(os.path.join(datapath, curr_folder, f))
        all_length.append(len(sub_folder))
        all_y.append(int(f.split('_')[0]))
        all_time.append(int(series_time.loc[series_time['target_folder'] ==f, ['series_time']].values[0][0]))
    return all_names, all_length, all_y, all_time

def get_name_length_d(datapath, datapath_d, curr_folder):
    '''
    A function to load DSA validation cases information 
    Parameters
    ----------
    datapath_d : TYPE, str
        DESCRIPTION. datapath of the folders to be analyzed, e.g. 'C:/Users/320190618/Documents/GUI_analysis' 
    curr_folder : TYPE, str 
        DESCRIPTION. specific folder to be analyzed e.g. 

    Returns
    -------
    all_names : TYPE, list
        DESCRIPTION. a list of folder names 
        e.g. [''C:/Users/320190618/Documents/GUI_analysis/DSA\\BIDMC-case2\\0_0', ...]
    all_length : TYPE, list 
        DESCRIPTION. a list of DICOM series lengths, [12, 10, 14, ...]
    all_y : TYPE, list 
        DESCRIPTION. a list of ground truth labels [0, 0, 0, 2, 2, 2, 2...]
    all_time : TYPE list 
        DESCRIPTION. a list of time points found in the DICOM header
    '''
    all_names = []
    all_length = []    # each video length
    all_y = []
    all_time = []
    # find series time 
    curr_case =  curr_folder.split('\\')[-1]
    df = pd.read_csv(os.path.join(datapath, 'dsa_' + curr_case + '.csv'), index_col=0)
    series_time = df.loc[df['target_test'].isna() == False, ['series_time', 'target_test']]
    series_time = series_time.drop_duplicates(subset=['target_test'], keep='first')
    # f names 
    frame_name  = os.listdir(os.path.join(datapath_d, curr_folder))
    for f in frame_name: # (4_0, 4_1, 4_2)
        all_names.append(os.path.join(datapath_d, curr_folder, f))
        sub_folder = os.listdir(os.path.join(datapath_d, curr_folder, f))
        all_length.append(len(sub_folder))
        all_y.append(int(f.split('_')[0]))
        all_time.append(int(series_time.loc[series_time['target_test'] ==f, ['series_time']].values[0][0]))
    return all_names, all_length, all_y, all_time


def get_eval_results(model_f, model_dsa, datapath, datapath_d, to_eval, to_eval_d, eval_ind, use_correction=True):
    '''
    A function to get validation results

    Parameters
    ----------
    model : TYPE, pytorch model
        DESCRIPTION, already loaded with trained weights
    datapath : TYPE, validation data path 
        DESCRIPTION. e.g. 'C:/Users/320190618/Documents/code_compile/2_gui/FEVAR_only'
    to_eval : TYPE, list 
        DESCRIPTION. list of all the validation records
    eval_ind : TYPE, ind
        DESCRIPTION, index of records being validated 
    use_correction : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    all_t : TYPE, list
        DESCRIPTION. list of timestamps
    all_y_pred : TYPE, list
        DESCRIPTION. list of model prediction 
    X_all : TYPE, list
        DESCRIPTION. list of all the image data 

    '''
    dev_names, dev_length, dev_y, dev_time = get_name_length(datapath, to_eval[eval_ind])
    dev_names_d, dev_length_d, dev_y_d, dev_time_d = get_name_length_d(datapath, datapath_d, to_eval_d[eval_ind])
    # dev sampler and dataloader
    dev_length_skip = skip_video (dev_length)
    # dev_list = list(zip(dev_names, dev_length, dev_time))
    select_frame = {'begin': 1, 'end': 100, 'skip': 1}
    dev_transform = transforms.Compose([transforms.Resize([512, 512]),
                                transforms.ToTensor()])

    #start validation 
    model_f.eval()
    test_loss = 0
    all_y, all_y_pred = [], []
    all_t = []
    X_all= []
    l_all = []
    
    infer_order = sorted(range(len(dev_time)), key=lambda k: dev_time[k])
    sorted_dev_names = [dev_names[i] for i in infer_order]
    sorted_dev_length =  [dev_length[i] for i in infer_order]
    sorted_dev_y = [dev_y[i] for i in infer_order]
    dev_time.sort()
    sorted_dev_list = list(zip(sorted_dev_names, sorted_dev_length, dev_time))
    dev_set  = Dataset(sorted_dev_list, sorted_dev_y, select_frame, transform=dev_transform)
    dev_loader = data.DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=col_fn)
    # pre_pred = torch.tensor([[0]])
    with torch.no_grad():
        for X, X_lengths, y, time_stamp in dev_loader:
        # distribute data to device
            X_lengths, y = X_lengths.view(-1, ), y.view(-1, )
    
            output = model_f(X)
            output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(X_lengths)])
    
            # loss = F.cross_entropy(output_true, y, reduction='sum')
            X_all.append(X)
            l_all.append(X_lengths)
            # test_loss += loss.item()                 # sum up minibatch loss
            y_pred = output_true.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-
                    
            # collect all y and y_pred in all batches
            all_y.append(y.numpy())
            all_y_pred.append(y_pred.numpy()[0][0])
            all_t.append(time_stamp.numpy())
        if use_correction == True:
            # s =  [all_y_pred[i][0][0] for i in range(len(all_y_pred))]
            ind = lis(all_y_pred)[3][::-1]
            ## hierachy tracking   
            ind_try = [2]*(len(all_y_pred) - len(ind))
            oor_ind = [i for i in range(len(all_y_pred)) if i not in ind]
            oor_dict = dict.fromkeys(oor_ind, 2)
            allowable = 0 
            while len(ind) < (len(all_y_pred) - allowable):   
                oor_ind = [i for i in range(len(all_y_pred)) if i not in ind]
               
                for i, oor in enumerate(oor_ind):
                    if (oor < (len(all_y_pred) - 1)) and ((all_y_pred[oor-1] in [2, 3]) and (all_y_pred[oor+1] in [2, 3])) and (all_y_pred[oor] in [2, 3]):
                        allowable += 1 
                        continue
                    output = model_f(X_all[oor])
                    output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(l_all[oor])])
                    all_y_pred[oor] = torch.topk(output_true, oor_dict[oor])[1][:, (oor_dict[oor] - 1)].numpy()[0]
                    oor_dict[oor] = oor_dict[oor] + 1 
                ind = lis(all_y_pred)[3][::-1] 
        # infer dsa 
        model_dsa.eval()
        test_loss = 0
        all_y_d, all_y_pred_d = [], []
        all_t_d = []
        X_all_d= []
        l_all_d = []
        infer_order_d = sorted(range(len(dev_time_d)), key=lambda k: dev_time_d[k])
        sorted_dev_names_d = [dev_names_d[i] for i in infer_order_d]
        sorted_dev_length_d =  [dev_length_d[i] for i in infer_order_d]
        sorted_dev_y_d = [dev_y_d[i] for i in infer_order_d]
        dev_time_d.sort()
        sorted_dev_list_d = list(zip(sorted_dev_names_d, sorted_dev_length_d, dev_time_d))
        dev_set_d  = Dataset(sorted_dev_list_d, sorted_dev_y_d, select_frame, transform=dev_transform)
        dev_loader_d = data.DataLoader(dev_set_d, batch_size=1, shuffle=False, collate_fn=col_fn)
        pre_pred = torch.tensor([[0]])
        with torch.no_grad():
            for X, X_lengths, y, time_stamp in dev_loader_d:
            # distribute data to device
                X_lengths, y = X_lengths.view(-1, ), y.view(-1, )
        
                output = model_dsa(X)
                output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(X_lengths)])
        
                # loss = F.cross_entropy(output_true, y, reduction='sum')
                X_all_d.append(X)
                l_all_d.append(X_lengths)
                # test_loss += loss.item()                 # sum up minibatch loss
                y_pred_d = output_true.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-
                # collect all y and y_pred in all batches
                all_y_d.append(y.numpy())
                all_y_pred_d.append(y_pred_d.numpy()[0][0])
                all_t_d.append(time_stamp.numpy())
        # roadmap label  3 can only by the last a few
        label4index = [i for i, k in enumerate(all_y_pred_d) if k==3]
        tailindex = len(all_y_pred_d) - 1
        for i, k in enumerate(label4index[::-1]):
            if k == tailindex:
                tailindex -= 1 
                label4index.remove(k)
                continue
            output = model_dsa(X_all_d[k])
            output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(l_all_d[k])])
            all_y_pred_d[k] = torch.topk(output_true, 2)[1][:, 1].numpy()[0]
    return all_t, all_y_pred, X_all, all_t_d, all_y_pred_d, X_all_d

def get_fors_results(fors_dir):
    '''
    Parameters
    ----------
    fors_datapath : TYPE, str
        DESCRIPTION. FORS data path e.g.'C:/Users/320190618/Documents/code_compile/2_gui/FEVAR_DSA_FORS/FORS'

    Returns
    -------
    time_keep : TYPE, list
        DESCRIPTION, list of timestamps after filtering using a 30-second threshold
    label_keep: TYPE, list
        DESCRIPTION. list of predictions converted using results_map

    '''
    results_map = {'Static': 0, 'Out of Body' :1, 'Navigation':2, 'Cannulation':3, 'Cannulated':4}
    df_total = pd.DataFrame({'Frame': [], 'Anatomical location': [], 'Prediction': [],  'Time':[]})
    for file in os.listdir(fors_dir):
        if file.endswith(".tsv"):
           df= pd.read_csv(os.path.join(fors_dir, file), sep='\t', names=['Frame', 'Anatomical location', 'Prediction', 'Time'])
           df['isStatusChanged'] = df['Prediction'].shift(1, fill_value=df['Prediction'].head(1)) != df['Prediction']
           # df_downsample =  df.
           df.iloc[0, 4] = True
           df_downsample = df.loc[df['isStatusChanged'] == True]
           df_total = pd.concat([df_total, df_downsample])
           
    df_total.sort_values(by=['Time'], inplace=True)
    timestamps_str= [i[:8] for i in df_total['Time'].values]
    timestamps_int = [str2int_time(i) for i in timestamps_str]
    prediction = [results_map[i] for i in df_total['Prediction'].values]
    gap = [timestamps_int[i] - timestamps_int[i-1] for i in range(1, len(timestamps_int))]
    index_to_keep = [i for i in range(len(gap)) if gap[i] > 30]
    index_to_keep.append(len(timestamps_int) -1)
    time_keep =  [timestamps_int[i] for i in index_to_keep]
    label_keep =  [prediction[i] for i in index_to_keep]
    return time_keep, label_keep

def str2int_time(str_time):
    '''
    Parameters
    ----------
    str_time : TYPE, str
        DESCRIPTION. original time stamps from shape_pred.tsv, e.g. '10:21:31.12000'

    Returns
    -------
    TYPE, str
        DESCRIPTION. only keep hour, minute and second, e.g. "10:21:31"

    '''
    hour = str_time[0:2]
    minute = str_time[3:5]
    second = str_time[6:8]
    return int(hour)*10000 + int(minute)*100 + int(second)

def diff_times(t1, t2):
    '''
    A funtion to compute time difference 

    Parameters
    ----------
    t1 : TYPE, str
        DESCRIPTION. starting time. e.g. '103128' or '10312.1122'
    t2 : TYPE, str
        DESCRIPTION. ending time,

    Returns
    -------
    TYPE, float
        DESCRIPTION. timediff in minutes 

    '''

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


def get_procedure_durations(all_t, all_y_pred, dataset_dict):
    '''

    Parameters
    ----------
    all_t : TYPE, list
        DESCRIPTION. list of time points from validation
    all_y_pred : TYPE, list 
        DESCRIPTION. list of prediction results from validation 
    dataset_dict: TYPE, dict
       DESCRIPTION, mean procedure duration in the dataset 

    Returns
    -------
    new_dict : TYPE, list 
        DESCRIPTION. list of procedure durations . e.g. ['Navigation: 5.8 min',
         'Sheath delivery: 7.1 min' ...]
    dataset_trim : TYPE, list 
        DESCRIPTION. list of dataset mean trimmed to contain only available procedures in this record  
    '''
    label_dict = {0: 'Navigation', 1: 'Sheath delivery', 2: 'Unsheathed', 3:'Cannulation', 4: 'Final deployment'}
    duration_dict = {}
    for j in range(5):
        duration_dict[j] = 0 
    duration_trim = []
    dataset_trim = []
    prev = all_y_pred[0]
    prev_time = str(all_t[0][0][0]) # get the value out and convert to str
    for i, pred in enumerate(all_y_pred[1:]): 
        if pred != prev: 
            duration_dict[prev] += diff_times(prev_time, str(all_t[i][0][0]))
            prev_time = str(all_t[i][0][0])
            prev = pred
        # deal with last label in the sequence 
        if i == len(all_y_pred) -2:
            diff =  diff_times(prev_time, str(all_t[i][0][0]))
            if diff>0:
                duration_dict[pred] += diff
    # only return procedure that's available:
    for i in range(5):
        if duration_dict[i] >0:
            duration_trim.append(label_dict[i] + ': %.1f' %duration_dict[i] + ' min')
            dataset_trim.append(dataset_dict[i])
    return duration_trim, dataset_trim