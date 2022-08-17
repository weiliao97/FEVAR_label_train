# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:53:49 2022

@author: 320190618
"""
# import os
import numpy as np
# from PIL import Image
# from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# import torchvision.transforms as transforms
# from tqdm import tqdm
# small resnet 

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

# ------------------------ CRNN module ---------------------- ##

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self, args, img_x=224, img_y=224, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet pretrained or not 
class ResCNNEncoder(nn.Module):
    def __init__(self, args, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        if args.encoder_model == 'res':
            resnet = models.resnet18(pretrained=args.pretrained)        
            modules = list(resnet.children())[:-1]      # delete the last fc layer.
    #        if args.img_size == 900:
    #          # (3, 900, 900) after head module will become (3, 224, 224)
    #          self.head =  nn.Sequential(
    #              nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
    #              nn.BatchNorm2d(3, momentum=0.01),
    #              nn.ReLU(),                      
    #              nn.MaxPool2d(kernel_size=2))
            self.resnet = nn.Sequential(*modules)
        elif args.encoder_model == 'res_s' or args.encoder_model == 'res_decode':
            self.resnet = res_s()
        # 256 since I made a custome resnet, 512 if using full resnet 
        if args.no_encode_fc == False:
          self.fc1 = nn.Linear(512, fc_hidden1)
          self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
          self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
          self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
          self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
          self.linear = nn.Linear(CNN_embed_dim, args.output_classes)
        self.no_decoder = args.no_decoder
        self.no_encode_fc = args.no_encode_fc
        
    def forward(self, x_3d):  # model grad input (4, 1, 512, 512)
#        cnn_embed_seq = []
        # x_3d can be (4, 100, 3, 224, 224)
#        for t in range(x_3d.size(1)):
            # ResNet CNN
            #with torch.no_grad():
#            if args.img_size == 900:
#              x = self.head(x_3d[:, t, :, :, :])
#              x = self.resnet(x)  # ResNet
#            else:
#            x = self.resnet(x_3d)   
        x = self.resnet(x_3d)  # ResNet  (4, 1, 512, 512)---> (4, 512, x, x)
        x = x.view(x.size(0), -1)             # flatten output of conv
        
        if self.no_encode_fc == False:
          # FC layers
          x = self.bn1(self.fc1(x))
          x = F.relu(x)
          x = self.bn2(self.fc2(x))
          x = F.relu(x)
          x = F.dropout(x, p=self.drop_p, training=self.training)
          x = self.fc3(x) 

#        cnn_embed_seq.append(x) # [(4, embed_dim), (4, embed_dim), ...]
            # x is ()

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
#        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  # (10, 4, features_dim)
        
        if self.no_decoder == True:
          x = self.linear(x)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return x # (4, 10, 5) 0r (4, 10, 64) 


class DecoderRNN_varlen(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN_varlen, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN, x_lengths):        
        N, T, n = x_RNN.size()
        # print('x_RNN.size:', x_RNN.size(), 'x_lengths:', x_lengths)

        for i in range(N):
            if x_lengths[i] < T:
                x_RNN[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_RNN.device)

        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_RNN[perm_idx], lengths_ordered, batch_first=True)
        self.LSTM.flatten_parameters()
        packed_RNN_out, (h_n_sorted, h_c_sorted) = self.LSTM(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))
        
        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        RNN_out = RNN_out[unperm_idx]

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of CRNN module ---------------------- ##
