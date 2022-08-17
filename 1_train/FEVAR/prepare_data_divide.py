"""
Created on Mon Jun 13 14:21:08 2022

@author: 320190618
"""
import os
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Sampler
import torch
import torch.nn.functional as F
import cv2 
from collections import defaultdict
import random 

# (968, 750), (750, 968)
# pad into equal 
def pad_img(img):
    '''
    A function to pad imaged into N by N
    Parameters
    ----------
    img : TYPE, unit8 image
        DESCRIPTION. original image, shape could be (968, 750)

    Returns
    -------
    padded_img : TYPE, unit8 image
        DESCRIPTION. padded image, shape becomes (968, 968), padding equally distributed on both sides of the short edge 

    '''
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1] # 968 - 750 = 218
        ld = int(diff/2)
        rd = diff - ld
        padded_img = cv2.copyMakeBorder(img, 0, 0, ld, rd, borderType= cv2.BORDER_CONSTANT)
    else:
        diff = img.shape[1] - img.shape[0]
        td = int(diff/2)
        bd = diff -td
        padded_img = cv2.copyMakeBorder(img, td, bd, 0, 0, cv2.BORDER_CONSTANT)
    return padded_img


class BySequenceLengthSampler(Sampler):
    
    '''
    A custome sampler to sample the dataset by length 

    Parameters
    ----------
    data_source : TYPE, list of tuple
        DESCRIPTION. e.g. [(original_index, length), (), () , ...]
    bucket_boundaries : TYPE, list 
        DESCRIPTION. separation intervals when grouping th data, e.g. [2, 8, 20, 100]
    batch_size : TYPE, optional, int
        DESCRIPTION. The default is 64.
    Returns
    -------
    None.
    '''
    def __init__(self, data_source, bucket_boundaries, batch_size=64):

        self.ind_n_len = data_source
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
        
    def __iter__(self):
        '''
        A funtion iterating through the dataset by length and putting the data into specific bucket
        '''
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.ind_n_len)
    
    def element_to_bucket_id(self, x, seq_length):
        '''
        
        Parameters
        ----------
        x : TYPE int
            DESCRIPTION. index in the dataset 
        seq_length : TYPE int
            DESCRIPTION. length of the image series 

        Returns
        -------
        bucket_id : TYPE, int 
            DESCRIPTION. which bucket a data belongs to

        '''
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id
    

# custom collate function to pad sequances in each batch based on by sequence sampler 

def col_fn(batchdata):
    '''
    custom collate function to pad sequances in each batch based on by sequence sampler 

    Parameters
    ----------
    batchdata : TYPE, list of tuples,
        DESCRIPTION.  each element is (img_data, length, label, timestamp), 
        e.g. [([length, 1, img_size, im_size], length, label, time_stamp), ()....]
    Returns
    -------
    TYPE, list of tuples
        DESCRIPTION. each elements is (padded_image_data, padded_length, padded_label, timestamp)

    '''
    # data items should be torch tensors already 
    len_data = len(batchdata)  
    # in batchdata, shape [(6, 3, 224, 224)]
    seq_len = [batchdata[i][0].shape[0] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    # len_tem = [np.zeros((batchdata[i][0].shape[0])) for i in range(len_data)]
    max_len = max(seq_len)

    # [(6, 3, 224, 224) ---> (8, 3, 224, 224)]
    padded_td = [F.pad(batchdata[i][0], pad=(0, 0, 0, 0, 0, 0, 0, max_len-batchdata[i][0].shape[0]), \
                   mode='constant', value=0) for i in range(len_data)]
    # iterate, length and label stay the smae 
    # [0, 1, 0, 0, 0, ...]
    padded_length = [batchdata[i][1] for i in range(len_data)]
    padded_label = [batchdata[i][2] for i in range(len_data)]
    
    return torch.stack(padded_td), torch.stack(padded_length), torch.stack(padded_label)


class Dataset(data.Dataset):

    def __init__(self, args, lists, labels, class_prob, transform=None, aug_transform=None):
        '''
        Parameters
        ----------
        args: from argparse
        lists : TYPE, list of data info
            DESCRIPTION. including validation folder name, image series length, ROI
        labels : TYPE, list
            DESCRIPTION, ground truth labels
        class_prob: TYPE, list
            DESCRIPTION, a list of the same fraction based on numbe rof classes
        transform : TYPE, optional
            DESCRIPTION. The default is None. 
        au_transform : TYPE, optional
            DESCRIPTION. Data augmentation transform 
        Returns
        -------
        None.

        '''
        # self.data_path = data_path
        self.labels = labels
        self.folders, self.video_len, self.frames, self.roi = list(zip(*lists))
        self.transform = transform
        self.resample = args.resample
        self.class_prob = class_prob
        self.items_per_class = defaultdict(list)
        for ii, c in enumerate(labels):
            self.items_per_class[c].append(ii)
        self.use_roi = args.use_roi
        self.aug_transform = aug_transform 

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)
    
    def __getitem__(self, index):
        
        "Generates one sample of data"
        # select sample
        selected_folder = self.folders[index]
        video_len = self.video_len[index]
        roi = self.roi[index]
        select = self.frames[index]
        img_size = self.transform.__dict__['transforms'][0].__dict__['size']       # get image resize from Transformation
        channels = 1

        # Load video frames
        X_padded = torch.zeros((len(select), channels, img_size[0], img_size[1]))   # input size: (frames, channels, image size x, image size y)
        # X_padded = []
        # if video len is 8, selected frame would be [1, 2, ...., 8]
        # if video len is large than 100, subsample by 2, if larger than 200, subsample by 4 
        for i, f in enumerate(select):
            frame = Image.open(os.path.join(selected_folder, '{:04d}.png'.format(f))) # f is 352, means 0352
            # while it's still PIL, Convert to (H, W, C)  
            frame_8 = cv2.convertScaleAbs(np.array(frame), alpha=(255.0/65535.0))
            if self.use_roi == True:
                frame_8 = frame_8[roi[0, 1] : roi[1, 1], roi[0, 0] : roi[1, 0]]
            # check size:
            if frame_8.shape[0] != frame_8.shape[1]:
                frame_8 = pad_img(frame_8)

            rgb_pil = Image.fromarray (frame_8, mode = 'L')
            frame = self.transform(rgb_pil) if self.transform is not None else frame  # impose transformation if exists
            X_padded[i, :] = frame 
        if self.aug_transform is not None:
            X_padded = self.aug_transform(X_padded)
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor
        video_len = torch.LongTensor([len(select)])

        return X_padded, video_len, y