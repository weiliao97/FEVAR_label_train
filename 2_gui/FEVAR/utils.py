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

def get_eval_results(model, datapath, to_eval, eval_ind, use_correction=True):
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
    # dev sampler and dataloader
    dev_length_skip = skip_video (dev_length)
    # dev_list = list(zip(dev_names, dev_length, dev_time))
    select_frame = {'begin': 1, 'end': 100, 'skip': 1}
    dev_transform = transforms.Compose([transforms.Resize([512, 512]),
                                transforms.ToTensor()])

    #start validation 
    model.eval()
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
    
            output = model(X)
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
                
                print('Perform correction...')
                oor_ind = [i for i in range(len(all_y_pred)) if i not in ind]
               
                for i, oor in enumerate(oor_ind):
                    if (oor < (len(all_y_pred) - 1)) and ((all_y_pred[oor-1] in [2, 3]) and (all_y_pred[oor+1] in [2, 3])) and (all_y_pred[oor] in [2, 3]):
                        allowable += 1 
                        continue
                    print(oor)
                    output = model(X_all[oor])
                    output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(l_all[oor])])
                    all_y_pred[oor] = torch.topk(output_true, oor_dict[oor])[1][:, (oor_dict[oor] - 1)].numpy()[0]
                    oor_dict[oor] = oor_dict[oor] + 1 
                ind = lis(all_y_pred)[3][::-1] 
    return all_t, all_y_pred, X_all