import numpy as np
import cv2
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

def check_center(shape, x, y, w, h):
    '''
    A funtion to check if the cropped region is in the center 
    Parameters
    ----------
    shape : TYPE, array 
        DESCRIPTION. image shape. e.g. [1024, 512]
    x : TYPE, int
        DESCRIPTION. bounding rect y coord
    y : TYPE, int
        DESCRIPTION. bounding rect x coord
    w : TYPE int 
        DESCRIPTION. bounding rect width 
    h : TYPE
        DESCRIPTION. bounding rect height 

    Returns
    -------
    bool
        DESCRIPTION. whether this bounding rect ceter is near image center

    '''
    x_min = shape[1]*0.45
    x_max = shape[1]*0.55
    y_min = shape[0]*0.45
    y_max = shape[0]*0.55
    if (x_min< (x+ w/2) < x_max) and (y_min < (y + h/2) < y_max):
        return True
    else:
        return False

def find_roi(image):
    '''
    A function to crop out unuseful black edges
    Parameters
    ----------
    image : TYPE, unit8 image
        DESCRIPTION. input image

    Returns
    -------
    TYPE, unit 8 cropped image
    '''
    level =  cv2.Laplacian(image, cv2.CV_8UC2).var()
    if image.shape[0] <1024:
        k_params = 3
    elif image.shape[0] <2048:
        k_params = 5
    else:
        k_params = 7
    if level > 3000: # blur
        gk_params = 19
    else:
        gk_params = 15
    if level > 3000: # blur
        gk_params = 19
    else:
        gk_params = 15
    blurred = cv2.GaussianBlur(image, (gk_params, gk_params), 0)
    canny =  cv2.Canny(blurred, 10, 40, 1)
    # Find contours2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_params, k_params))
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    opening =  cv2.dilate(canny, kernel, iterations = 5)
    opening[0, :] = 1
    opening[-1, : ] = 1
    opening[:, 0 ] = 1
    opening[:, -1] = 1
    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Iterate thorugh contours and draw rectangles around contours
    rect_area = []
    rects = []
    for c in cnts:
        area = cv2.contourArea(c)/(image.shape[0] * image.shape[1])
        if area > 0.1:
            x,y,w,h = cv2.boundingRect(c)    
            is_center = check_center(image.shape, x, y, w, h)
            if is_center == True:
                rect_area.append(area)
                rects.append((x, y, w, h))
        
    min_area = min(rect_area)
    min_index = rect_area.index(min_area)
    (x, y, w, h) = rects[min_index]
    return np.asarray([[y, y+h], [x, x+w]])


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
    def __init__(self, args, lists, labels, set_frame, class_prob, transform=None, aug_transform=None):
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
        set_frame : TYPE, dictionary
            DESCRIPTION, frames to begin and to end, almost always use the default 
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
        self.folders, self.video_len= list(zip(*lists))
        self.set_frame = set_frame
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
        
        def getskip(length):
            if length >= 100 and length < 200:
                return 2
            elif length >=200:
                return 4
            else:
                return 1 
        "Generates one sample of data"
        # select sample
        selected_folder = self.folders[index]
        video_len = self.video_len[index]
        select = np.arange(self.set_frame['begin'], video_len + 1, getskip(video_len))
        img_size = self.transform.__dict__['transforms'][0].__dict__['size']       # get image resize from Transformation
        channels = 1
        # Load video frames
        X_padded = torch.zeros((len(select), channels, img_size[0], img_size[1]))   # input size: (frames, channels, image size x, image size y)
        # X_padded = []
        # if video len is 8, selected frame would be [1, 2, ...., 8]
        # if video len is large than 100, subsample by 2, if larger than 200, subsample by 4 
        for i, f in enumerate(select):
            frame = Image.open(os.path.join(selected_folder, '{:04d}.png'.format(f-1)))
            # while it's still PIL, Convert to (H, W, C)  
            frame_8 = cv2.convertScaleAbs(np.array(frame), alpha=(255.0/65535.0))
            if i == 0:
              roi = find_roi(frame_8)
            if self.use_roi == True:
                frame_8 = frame_8[roi[0, 0] : roi[0, 1], roi[1, 0] : roi[1, 1]]
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