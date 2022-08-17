import torchvision.models as models
import torch
import torch.nn as nn

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
#%%
import argparse
import torch.nn as nn
import torch 
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import accuracy_score
import torch.nn.functional as F 
import numpy as np 
import pandas as pd
import os
import prepare_data_divide
import model
import random 
from collections import defaultdict
#os.environ['CUDA_VISIBLE_DEVICES']='1'

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU\

## by sequence sampler
#len_range = [i for i in range(11)]
#
## split into the required sequence, for example, 82, 9 returns [9, 9, 9, 9, 9, 9, 9, 9, 10]
#def split(x, n):
#    seg = []
# 
#    # If x % n == 0 then the minimum
#    # difference is 0 and all
#    # numbers are x / n
#    if (x % n == 0):
#        for i in range(n):
#            seg.append(x//n)
#    else:
#        # upto n-(x % n) the values
#        # will be x / n
#        # after that the values
#        # will be x / n + 1
#        zp = n - (x % n)
#        pp = x//n
#        for i in range(n):
#            if(i>= zp):
#                seg.append(pp+1)
#            else:
#                seg.append(pp)
#    return seg 
#    
#def get_name_length(datapath, all_folders, index):
#    roi = pd.read_pickle(os.path.join(datapath, 'crop.pkl'))
#    all_names = []
#    all_length = []    # each video length, sunsampled or skipped
#    all_frames = []   # which frame to read
#    all_roi = [] # roi in this 
#    all_y = []
#    # f names 
#    for ind in index:
#        curr_folder = all_folders[ind]
#        frame_name  = os.listdir(os.path.join(datapath, curr_folder))
#        for f in frame_name: # (4_0, 4_1, 4_2)
#            roi_f = roi.loc[(roi['Folder'] == curr_folder) & (roi['Cat'] == f), 'ROI'].values[0]# get an array, curr_fodler can be BIDMC-case1 
#            
#            sub_folder = os.listdir(os.path.join(datapath, curr_folder, f)) # 0000, 0001, 0002, 0003, 
#            l = len(sub_folder)
#            if l > 200: 
#                l_skip = int((l+1)/4)
#                l_seg = int(l_skip/10) + 1 
#                seg = split(l_skip, l_seg) # [9, 9, 9, 10]
#                seg_cum = np.cumsum(seg) # [9, 18, 27, 37]
#                for s_ind, s in enumerate(seg_cum): # s is 9, 18, 27, 37
#                    all_names.append(os.path.join(datapath, curr_folder, f))
#                    all_length.append(seg[s_ind])
#                    if s_ind == 0:
#                      all_frames.append(list(range(0, s*4, 4)))
#                    else: 
#                      all_frames.append(list(range(seg_cum[s_ind-1]*4, s*4, 4)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
#                    all_roi.append(roi_f)
#                    all_y.append(int(f.split('_')[0]))
#              
#            elif l > 100:
#                l_skip = int((l+1)/2)
#                l_seg = int(l_skip/10) + 1 
#                seg = split(l_skip, l_seg) # [9, 9, 9, 10]
#                seg_cum = np.cumsum(seg) # [9, 18, 27, 37]
#                for s_ind, s in enumerate(seg_cum): # s is 9, 18, 27, 37
#                    all_names.append(os.path.join(datapath, curr_folder, f))
#                    all_length.append(seg[s_ind])
#                    if s_ind == 0:
#                      all_frames.append(list(range(0, s*2, 2)))
#                    else: 
#                      all_frames.append(list(range(seg_cum[s_ind-1]*2, s*2, 2)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
#                    all_roi.append(roi_f)
#                    all_y.append(int(f.split('_')[0]))
#            
#            elif l > 10: # split (11, 2) = [5, 6]
#                l_seg = int(l/10) + 1 
#                seg = split(l, l_seg) 
#                seg_cum = np.cumsum(seg) # [5, 11]
#                for s_ind, s in enumerate(seg_cum): 
#                    all_names.append(os.path.join(datapath, curr_folder, f))
#                    all_length.append(seg[s_ind])
#                    if s_ind == 0:
#                      all_frames.append(list(range(0, s)))
#                    else: 
#                      all_frames.append(list(range(seg_cum[s_ind-1], s)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
#                    all_roi.append(roi_f)
#                    all_y.append(int(f.split('_')[0]))
#            
#            else: 
#                all_names.append(os.path.join(datapath, curr_folder, f))
#                all_length.append(l)
#                all_frames.append(list(range(0, l)))
#                all_roi.append(roi_f)
#                all_y.append(int(f.split('_')[0]))
#                # only append once 
#    return all_names, all_length, all_frames, all_roi, all_y
#
#def generate_buckets(bs, train_hist):
#    buckets = []
#    sum = 0
#    s = 0
#    buckets.append(2)
#    for i in range(10): 
#        # train_hist[0] is len [0, 1), train_hist[354] is  [354, 355]
#        sum +=train_hist[i] 
#        if sum>bs:
#            buckets.append(i+1)
#            sum = 0 
#    # residue is 58 < 128, remove index 205, attach 217, largest is 216
#    if sum < bs:
#        buckets.pop(-1)    
#    buckets.append(11)
#    return buckets
#
## update train_length  
#def skip_video(train_length):
#    skip = []
#    for i in train_length:
#        if i >=100 and i < 200:
#            skip.append(int((i+1)/2))
#        elif i>=200: 
#            skip.append(int((i+1)/4))
#        else: 
#            skip.append(i)
#    return skip 
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Process some integers.')
#    # dataset parameters
#    parser.add_argument("--dataset_path", type = str, default = '/local/home/320190618/FEVAR_dataset', help = "Path to the image folders") 
#    # model parameters
#    parser.add_argument("--img_size", type = int, default = 224, help = "Image size in the model") 
#    # EncoderCNN architecture
#    parser.add_argument('--encoder_model', default='convnet', const='convnet', nargs='?', choices=['res', 'res_s', 'convnet'])
#    parser.add_argument("--no_decoder", action = 'store_true', default=False, help="whether use rnn") 
#    parser.add_argument("--pretrained", action = 'store_true', default=False, help="whether use pretrained models") 
#    parser.add_argument("--CNN_fc_hidden1", type = int, default = 256)
#    parser.add_argument("--CNN_fc_hidden2", type = int, default = 128)
#    parser.add_argument("--CNN_embed_dim", type = int, default = 64, help = "latent dim extracted by 2D CNN")
#    parser.add_argument("--dropout_p", type = float, default = 0.2, help = "dropout probability")
#    # DecoderRNN architecture
#    parser.add_argument("--RNN_hidden_layers", type = int, default = 2)
#    parser.add_argument("--RNN_hidden_nodes", type = int, default = 128)
#    parser.add_argument("--RNN_FC_dim", type = int, default = 64)
#    parser.add_argument("--output_classes", type = int, default = 5)
#    # dataloader parameters
#    parser.add_argument("--resample", action = 'store_true', default=False, help="whether use dataset resampling") 
#    parser.add_argument("--reweight", action = 'store_true', default=False, help="whether use dataset reweighting") 
#    parser.add_argument("--use_roi", action = 'store_true', default=False, help="whether use roi cropping")
#    parser.add_argument("--data_aug", action = 'store_true', default=False, help="whether use data augmentation on train") 
#    parser.add_argument("--max_frame", type = int, default = 355, help = "Max frame number in a clip") 
#    parser.add_argument("--bucket_size", type = int, default = 20, help = "Bucket size to sort the train data by length")
#    # training parameters
#    parser.add_argument("--batch_size", type = int, default = 4)
#    parser.add_argument("--learning_rate", type = float, default = 1e-4)
#    parser.add_argument("--epochs", type = int, default = 150)
#    parser.add_argument("--patience", type = int, default = 20, help="Patient for early stopping")
#    #checkpoint
#    parser.add_argument("--save_model_path", type=str, default='/local/home/320190618/checkpoints', help = 'path to save the model')
#    parser.add_argument("--checkpoint_name", type=str, default='test', help = 'name to log the weights')
#    
#    args = parser.parse_args()
#    # make a folder name checkpoint_name to log loss acc and weights
##    ck_path = os.path.join(args.save_model_path, args.checkpoint_name)
##    if not os.path.isdir(ck_path):
##        os.mkdir(ck_path)
#    # 5 classes for now. This is for resmapling only 
#    class_prob = [1/5 for _ in range(5)]
#    # trainindex and validation index 
#    filenames= os.listdir (args.dataset_path) # get all files' and folders' names in the current directory
#    all_folders = []
#    for filename in filenames: # loop through all the files and folders
#        if os.path.isdir(os.path.join(args.dataset_path, filename)): # check whether the current object is a folder or not
#           all_folders.append(filename)
#    all_folders.sort()
#    # subject to change 
#    dev_index = [2, 6]
#    train_index = [i for i in range(16) if i not in dev_index]
#    train_names, train_length, train_frames, train_roi, train_y = get_name_length(args.dataset_path, all_folders, train_index)
#    dev_names, dev_length, dev_frames, dev_roi, dev_y = get_name_length(args.dataset_path, all_folders, dev_index)
#    print(dev_names)
#    train_list = list(zip(train_names, train_length, train_frames, train_roi))
#    dev_list = list(zip(dev_names, dev_length, dev_frames, dev_roi))
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
#        out = out.view(out.size(0), -1)
#        out = self.linear(out)
        return out


def res_s():
    return ResNet(BasicBlock, [2, 2, 2, 2])
  
  
    
model = res_s()
#modules = list(resnet.children())[:8]
#print(len(list(resnet.children())))
#for i in range(10):
#    print('next')
#    print(list(resnet.children())[i])
#model = nn.Sequential(*modules)
inp = torch.rand(1, 1, 512, 512)
out = model(inp)
print(out.shape)
##print(model.layer4[-1])