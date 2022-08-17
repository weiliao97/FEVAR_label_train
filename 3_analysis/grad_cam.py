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
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU\

# by sequence sampler
len_range = [i for i in range(11)]

# split into the required sequence, for example, 82, 9 returns [9, 9, 9, 9, 9, 9, 9, 9, 10]
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
    
def get_name_length(datapath, all_folders, index):
    roi = pd.read_pickle(os.path.join(datapath, 'crop.pkl'))
    all_names = []
    all_length = []    # each video length, sunsampled or skipped
    all_frames = []   # which frame to read
    all_roi = [] # roi in this 
    all_y = []
    # f names 
    for ind in index:
        curr_folder = all_folders[ind]
        frame_name  = os.listdir(os.path.join(datapath, curr_folder))
        for f in frame_name: # (4_0, 4_1, 4_2)
            roi_f = roi.loc[(roi['Folder'] == curr_folder) & (roi['Cat'] == f), 'ROI'].values[0]# get an array, curr_fodler can be BIDMC-case1 
            
            sub_folder = os.listdir(os.path.join(datapath, curr_folder, f)) # 0000, 0001, 0002, 0003, 
            l = len(sub_folder)
            if l > 200: 
                l_skip = int((l+1)/4)
                l_seg = int(l_skip/10) + 1 
                seg = split(l_skip, l_seg) # [9, 9, 9, 10]
                seg_cum = np.cumsum(seg) # [9, 18, 27, 37]
                for s_ind, s in enumerate(seg_cum): # s is 9, 18, 27, 37
                    all_names.append(os.path.join(datapath, curr_folder, f))
                    all_length.append(seg[s_ind])
                    if s_ind == 0:
                      all_frames.append(list(range(0, s*4, 4)))
                    else: 
                      all_frames.append(list(range(seg_cum[s_ind-1]*4, s*4, 4)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
                    all_roi.append(roi_f)
                    all_y.append(int(f.split('_')[0]))
              
            elif l > 100:
                l_skip = int((l+1)/2)
                l_seg = int(l_skip/10) + 1 
                seg = split(l_skip, l_seg) # [9, 9, 9, 10]
                seg_cum = np.cumsum(seg) # [9, 18, 27, 37]
                for s_ind, s in enumerate(seg_cum): # s is 9, 18, 27, 37
                    all_names.append(os.path.join(datapath, curr_folder, f))
                    all_length.append(seg[s_ind])
                    if s_ind == 0:
                      all_frames.append(list(range(0, s*2, 2)))
                    else: 
                      all_frames.append(list(range(seg_cum[s_ind-1]*2, s*2, 2)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
                    all_roi.append(roi_f)
                    all_y.append(int(f.split('_')[0]))
            
            elif l > 10: # split (11, 2) = [5, 6]
                l_seg = int(l/10) + 1 
                seg = split(l, l_seg) 
                seg_cum = np.cumsum(seg) # [5, 11]
                for s_ind, s in enumerate(seg_cum): 
                    all_names.append(os.path.join(datapath, curr_folder, f))
                    all_length.append(seg[s_ind])
                    if s_ind == 0:
                      all_frames.append(list(range(0, s)))
                    else: 
                      all_frames.append(list(range(seg_cum[s_ind-1], s)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ]
                    all_roi.append(roi_f)
                    all_y.append(int(f.split('_')[0]))
            
            else: 
                all_names.append(os.path.join(datapath, curr_folder, f))
                all_length.append(l)
                all_frames.append(list(range(0, l)))
                all_roi.append(roi_f)
                all_y.append(int(f.split('_')[0]))
                # only append once 
    return all_names, all_length, all_frames, all_roi, all_y

def generate_buckets(bs, train_hist):
    buckets = []
    sum = 0
    s = 0
    buckets.append(2)
    for i in range(10): 
        # train_hist[0] is len [0, 1), train_hist[354] is  [354, 355]
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i+1)
            sum = 0 
    # residue is 58 < 128, remove index 205, attach 217, largest is 216
    if sum < bs:
        buckets.pop(-1)    
    buckets.append(11)
    return buckets

# update train_length  
def skip_video(train_length):
    skip = []
    for i in train_length:
        if i >=100 and i < 200:
            skip.append(int((i+1)/2))
        elif i>=200: 
            skip.append(int((i+1)/4))
        else: 
            skip.append(i)
    return skip 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # dataset parameters
    parser.add_argument("--dataset_path", type = str, default = '/local/home/320190618/FEVAR_dataset', help = "Path to the image folders") 
    # model parameters
    parser.add_argument("--img_size", type = int, default = 224, help = "Image size in the model") 
    # EncoderCNN architecture
    parser.add_argument('--encoder_model', default='convnet', const='convnet', nargs='?', choices=['res', 'res_s', 'convnet'])
    parser.add_argument("--no_decoder", action = 'store_true', default=False, help="whether use rnn") 
    parser.add_argument("--pretrained", action = 'store_true', default=False, help="whether use pretrained models") 
    parser.add_argument("--CNN_fc_hidden1", type = int, default = 256)
    parser.add_argument("--CNN_fc_hidden2", type = int, default = 128)
    parser.add_argument("--CNN_embed_dim", type = int, default = 64, help = "latent dim extracted by 2D CNN")
    parser.add_argument("--dropout_p", type = float, default = 0.2, help = "dropout probability")
    # DecoderRNN architecture
    parser.add_argument("--RNN_hidden_layers", type = int, default = 2)
    parser.add_argument("--RNN_hidden_nodes", type = int, default = 128)
    parser.add_argument("--RNN_FC_dim", type = int, default = 64)
    parser.add_argument("--output_classes", type = int, default = 5)
    # dataloader parameters
    parser.add_argument("--resample", action = 'store_true', default=False, help="whether use dataset resampling") 
    parser.add_argument("--reweight", action = 'store_true', default=False, help="whether use dataset reweighting") 
    parser.add_argument("--use_roi", action = 'store_true', default=False, help="whether use roi cropping")
    parser.add_argument("--max_frame", type = int, default = 355, help = "Max frame number in a clip") 
    parser.add_argument("--bucket_size", type = int, default = 20, help = "Bucket size to sort the train data by length")
    # training parameters
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--epochs", type = int, default = 150)
    parser.add_argument("--patience", type = int, default = 20, help="Patient for early stopping")
    #checkpoint
    parser.add_argument("--save_model_path", type=str, default='/local/home/320190618/checkpoints', help = 'path to save the model')
    parser.add_argument("--checkpoint_name", type=str, default='test', help = 'name to log the weights')
    parser.add_argument("--weights_name", type=str, default='acc_epoch11.pth', help = 'name to log the weights')
    
    args = parser.parse_args()
    # make a folder name checkpoint_name to log loss acc and weights
    ck_path = os.path.join(args.save_model_path, args.checkpoint_name)
    if not os.path.isdir(ck_path):
        os.mkdir(ck_path)
    # 5 classes for now. This is for resmapling only 
    class_prob = [1/5 for _ in range(5)]
    # trainindex and validation index 
    filenames= os.listdir (args.dataset_path) # get all files' and folders' names in the current directory
    all_folders = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(args.dataset_path, filename)): # check whether the current object is a folder or not
           all_folders.append(filename)
    all_folders.sort()
    # subject to change 
    dev_index = [0, 8]
    train_index = [i for i in range(16) if i not in dev_index]
    train_names, train_length, train_frames, train_roi, train_y = get_name_length(args.dataset_path, all_folders, train_index)
    dev_names, dev_length, dev_frames, dev_roi, dev_y = get_name_length(args.dataset_path, all_folders, dev_index)
    
    train_list = list(zip(train_names, train_length, train_frames, train_roi))
    dev_list = list(zip(dev_names, dev_length, dev_frames, dev_roi))
    # create model
    if args.encoder_model == 'res' or args.encoder_model == 'res_s': 
        cnn_encoder = model.ResCNNEncoder(args, fc_hidden1=args.CNN_fc_hidden1, fc_hidden2=args.CNN_fc_hidden2, 
                            drop_p=args.dropout_p, CNN_embed_dim=args.CNN_embed_dim).to(device)
    elif args.encoder_model =='convnet': 
        cnn_encoder = model.EncoderCNN(args, img_x = args.img_size, img_y = args.img_size, fc_hidden1=args.CNN_fc_hidden1, fc_hidden2=args.CNN_fc_hidden2, 
                            drop_p=args.dropout_p, CNN_embed_dim=args.CNN_embed_dim).to(device)
    else:
      raise Exception("Error!, encoder not implemented!")
    # create dataloaders
    dev_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]),
                                transforms.ToTensor(),
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    train_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]),
                                transforms.ToTensor(),
#                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    select_frame = {'begin': 1, 'end': args.max_frame, 'skip': 1}
#    train_set  = prepare_data.Dataset(args, train_list, train_y, select_frame, class_prob, stage = 'training', transform=transform)
#    train_loader = data.DataLoader(train_set, batch_size= args.batch_size, num_workers = 1, pin_memory = True)
    # validation phase, no resampling needed 
#    dev_set  = prepare_data.Dataset(args, dev_list, dev_y, select_frame, class_prob, stage = 'eval', transform=transform)
#    dev_loader = data.DataLoader(dev_set, batch_size= args.batch_size, num_workers = 1, pin_memory = True)
##    
    # start training 
    best_loss = 1e4
    best_acc = 0.2
    patience_log = 0
    log_interval = 10
    # loss and opt
    ce_loss = nn.CrossEntropyLoss()
    crnn_params = list(cnn_encoder.parameters())
    # print trainable params
    cnn_params = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)
#    rnn_params = sum(p.numel() for p in rnn_decoder.parameters() if p.requires_grad)
    print('\nCNN trainable params are: %d\n'%cnn_params)
    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_dev_losses = []
    epoch_dev_scores = []
    # dev sampler 
    dev_hist, _ = np.histogram(dev_length, bins=len_range) # (0, 0, 19, 7, 2, 6, 7, 2, 6, 14)
    devbucket_boundaries = generate_buckets(8, dev_hist)  # (2, 3, 5, 7, 11)
    
    dev_pack = list(zip([i for i in range(len(dev_length))], dev_length))
    dev_sampler = prepare_data_divide.BySequenceLengthSampler(dev_pack, devbucket_boundaries, args.batch_size)
    dev_set  = prepare_data_divide.Dataset(args, dev_list, dev_y, class_prob, transform=dev_transform)
    dev_loader = data.DataLoader(dev_set, batch_size=1, batch_sampler=dev_sampler, collate_fn=prepare_data_divide.col_fn)
    
     # load model weights
    cnn_encoder.load_state_dict(torch.load(os.path.join(args.save_model_path, args.checkpoint_name, 'cnn_encoder_'+args.weights_name)))
    # do some inference 
    #start validation 
    test_loss = 0
    all_y, all_y_pred = [], []
    l_list = []
    print(cnn_encoder.training)
    target_layers = [cnn_encoder.resnet.layer4[-1]] # fore resnet 18 
    cam = GradCAM(model=cnn_encoder, target_layers=target_layers, use_cuda=use_cuda) # model output is (4, 10, 5)
    
    image_array = []
    cam_array = []
    for X, X_lengths, y in dev_loader: 
        # distribute data to device
        # X can be [4, 10, 1, 512, 512), y is [4] 
        cnn_encoder.train()
        X, X_lengths, y = X.to(device), X_lengths.view(-1, ), y.to(device).view(-1, )

        for f in range(X.shape[1]):
            temp = X[:, f]
            grayscale_cam = cam(input_tensor=temp) # if input is (4, 1, 512, 512), the output is (4, 1, 5)
            print(grayscale_cam.shape)
            image_array.append(temp.detach().cpu().numpy()) #(4, 512, 512)
            cam_array.append(grayscale_cam)
        # do some eval 
        cnn_encoder.eval()
        output = cnn_encoder(X[:, -1]) # (4, 5)
#        output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(X_lengths)])

#        loss = F.cross_entropy(output_true, y, reduction='sum')
#        test_loss += loss.item()                 # sum up minibatch loss
        y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

        # collect all y and y_pred in all batches
        all_y.extend(y)
        all_y_pred.extend(y_pred)
        l_list.append(len(y))
    
        
    image_array = np.concatenate(image_array, axis=0)
    cam_array = np.concatenate(cam_array, axis=0)
    np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'cam_t.npy'), cam_array)
    np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'image_t.npy'), image_array)
        
#          grayscale_cam = grayscale_cam[0, :]
#        print(grayscale_cam.shape)


#
#    test_loss /= len(dev_loader.dataset)
#
#    # compute accuracy
    all_y = torch.stack(all_y, dim=0).detach().cpu().numpy()
    all_y_pred = torch.stack(all_y_pred, dim=0).detach().cpu().numpy()
    np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'y.npy'), all_y)
    np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'y_pred.npy'), all_y_pred)
    np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'length_map.npy'), np.asarray(l_list))
#    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
#    
#    # show information
#    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
#    print(all_y)
#    print(all_y_pred)
#    dev_loss = np.load(os.path.join(args.save_model_path, args.checkpoint_name, 'dev_loss.npy'))
#    dev_acc = np.load(os.path.join(args.save_model_path, args.checkpoint_name, 'dev_score.npy'))
#    print(dev_loss)
#    print(dev_acc)
#    train_loss = np.load(os.path.join(args.save_model_path, args.checkpoint_name, 'training_loss.npy'))
#    train_acc = np.load(os.path.join(args.save_model_path, args.checkpoint_name, 'training_score.npy'))
#    print(train_loss)
#    print(train_acc)    