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

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# by sequence sampler
len_range = [i for i in range(11)]

# split into the required sequence, for example, 82, 9 returns [9, 9, 9, 9, 9, 9, 9, 9, 10]
def split(x, n):
    '''
    A function to find how to split long sequences
    args:
      x: type: int, description: total length of the DICOM series
      n: type: int, description: number of small segments 
    return:
      seg : type: list, description: split results
    '''
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
    '''
    A function to load train/validation cases information 
    Parameters
    ----------
    datapath : TYPE, str
        DESCRIPTION. datapath of the folders to be analyzed, e.g. 'C:/Users/320190618/Documents/code_compile/1_training/FEVAR/FEVAR_dataset' 
    all_folder : TYPE, list
        DESCRIPTION. all the train/validation folders to be analyzed  
    index, TYPE, list
        DESCRIPTION. indiced for train e.g.[0, 2, 3, ...] or val [1, 6, ...]
    Returns
    -------
    all_names : TYPE, list
        DESCRIPTION. a list of folder names 
        e.g. [''C:/Users/320190618/Documents/code_compile/1_training/FEVAR/FEVAR_dataset\\BIDMC-case2\\0_0', ...]
    all_length : TYPE, list 
        DESCRIPTION. a list of DICOM series lengths, [12, 10, 14, ...]
    all_frame: TYPE, list of frame indices to be loaded
        DESCRIPTION. e.g. [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], ....]
    all_roi : TYPE list 
        DESCRIPTION. a list of ROI for each series e.g. [[[100, 100], [1004, 1004]], ..]
    all_y: TYPE list
        DESCRIPTION. a list of ground truth labels

    '''
    # In this task, ROI is pre-located by a script and saved in crop.pkl. In DSA task, ROI is found by when loading data
    roi = pd.read_pickle(os.path.join(datapath, 'crop.pkl'))
    all_names = []
    all_length = []    # each video length, subsampled or skipped
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
            # if DICOM series length is larger than 200, then it does subsampling by 4 and then split into small sequences to save GPU memory
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
                      all_frames.append(list(range(seg_cum[s_ind-1]*4, s*4, 4)))  # 9 will be turned into  [0, 4, 8, 12, 16, 20, 24, 28, 32] , next will be [36, 40, 44, ...]
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
                      all_frames.append(list(range(seg_cum[s_ind-1]*2, s*2, 2))) 
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
                      all_frames.append(list(range(seg_cum[s_ind-1], s))) 
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
    buckets.append(2)
    for i in range(10): 
        # train_hist[0] is len [0, 1), train_hist[354] is  [354, 355]
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i+1)
            sum = 0 
    if sum < bs:
        buckets.pop(-1)    
    buckets.append(11)
    return buckets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # dataset parameters
    parser.add_argument("--dataset_path", type = str, default = '/local/home/320190618/FEVAR_dataset', help = "Path to the image folders") 
    parser.add_argument('--val_set', nargs='+', type=int)
    # model parameters
    parser.add_argument("--img_size", type = int, default = 512, help = "Image size in the model") 
    # EncoderCNN architecture
    parser.add_argument('--encoder_model', default='convnet', const='convnet', nargs='?', choices=['res', 'convnet'])
    parser.add_argument("--no_decoder", action = 'store_true', default=False, help="whether use rnn")  
    parser.add_argument("--no_encode_fc", action = 'store_true', default=False, help="whether having fc layers after CNN") 
    parser.add_argument("--pretrained", action = 'store_true', default=False, help="whether use pretrained models") 
    parser.add_argument("--CNN_fc_hidden1", type = int, default = 256, help = "fc layer dimention")
    parser.add_argument("--CNN_fc_hidden2", type = int, default = 128, help = "fc layer dimention")
    parser.add_argument("--CNN_embed_dim", type = int, default = 64, help = "latent dim extracted by 2D CNN")
    parser.add_argument("--dropout_p", type = float, default = 0.2, help = "dropout probability")
    # DecoderRNN architecture, if used 
    parser.add_argument("--RNN_hidden_layers", type = int, default = 2)
    parser.add_argument("--RNN_hidden_nodes", type = int, default = 128)
    parser.add_argument("--RNN_FC_dim", type = int, default = 64)
    parser.add_argument("--output_classes", type = int, default = 5)
    # dataloader parameters
    parser.add_argument("--resample", action = 'store_true', default=False, help="whether use dataset resampling") 
    parser.add_argument("--reweight", action = 'store_true', default=False, help="whether use dataset reweighting") 
    parser.add_argument("--use_roi", action = 'store_true', default=False, help="whether use roi cropping")
    parser.add_argument("--data_aug", action = 'store_true', default=False, help="whether use data augmentation on train") 
    parser.add_argument("--max_frame", type = int, default = 355, help = "Max frame number in a clip") 
    parser.add_argument("--bucket_size", type = int, default = 20, help = "Bucket size to sort the train data by length")
    # training parameters
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--epochs", type = int, default = 150)
    parser.add_argument("--patience", type = int, default = 20, help="Patience for early stopping")
    #checkpoint
    parser.add_argument("--save_model_path", type=str, default='/local/home/320190618/checkpoints', help = 'path to save the model')
    parser.add_argument("--checkpoint_name", type=str, default='test', help = 'name to log the weights')
    
    args = parser.parse_args()
    # make a folder name checkpoint_name to log loss acc and weights
    ck_path = os.path.join(args.save_model_path, args.checkpoint_name)
    if not os.path.isdir(ck_path):
        os.mkdir(ck_path)
    # 5 classes for now. This is for resampling only 
    class_prob = [1/args.output_classes for _ in range(args.output_classes)]
    # trainindex and validation index 
    filenames= os.listdir (args.dataset_path) # get all files' and folders' names in the current directory
    all_folders = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(args.dataset_path, filename)): # check whether the current object is a folder or not
           all_folders.append(filename)
    all_folders.sort()
    # subject to change 
    dev_index = args.val_set
    train_index = [i for i in range(len(all_folders)) if i not in dev_index]
    train_names, train_length, train_frames, train_roi, train_y = get_name_length(args.dataset_path, all_folders, train_index)
    dev_names, dev_length, dev_frames, dev_roi, dev_y = get_name_length(args.dataset_path, all_folders, dev_index)
    
    train_list = list(zip(train_names, train_length, train_frames, train_roi))
    dev_list = list(zip(dev_names, dev_length, dev_frames, dev_roi))
    # create model
    if args.encoder_model == 'res': 
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
                                ])
    train_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]),
                                transforms.ToTensor(),
                                ])
    # use RandomApply so the rotation is not always done, consistant with the rest 
    aug_transform = transforms.Compose([transforms.RandomApply(torch.nn.ModuleList([
     transforms.RandomRotation((-30, 30)), ]), p=0.5), 
     transforms.RandomPerspective(distortion_scale=0.2),
     transforms.RandomAdjustSharpness(sharpness_factor=2),
     transforms.RandomAutocontrast(),
     transforms.RandomHorizontalFlip()
    ]) if args.data_aug == True else None 
                                
    select_frame = {'begin': 1, 'end': args.max_frame, 'skip': 1}   
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
    model_opt = torch.optim.Adam(crnn_params, lr=args.learning_rate)
    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_dev_losses = []
    epoch_dev_scores = []
    # dev sampler 
    dev_hist, _ = np.histogram(dev_length, bins=len_range) # e.g. [0, 0, 19, 7, 2, 6, 7, 2, 6, 14]
    devbucket_boundaries = generate_buckets(8, dev_hist)  #  e.g. [2, 3, 5, 7, 11]
    
    dev_pack = list(zip([i for i in range(len(dev_length))], dev_length))
    dev_sampler = prepare_data_divide.BySequenceLengthSampler(dev_pack, devbucket_boundaries, args.batch_size)
    dev_set  = prepare_data_divide.Dataset(args, dev_list, dev_y, class_prob, transform=dev_transform)
    dev_loader = data.DataLoader(dev_set, batch_size=1, batch_sampler=dev_sampler, collate_fn=prepare_data_divide.col_fn)
    # generate a train class dict 
    items_per_class = defaultdict(list)
    for ii, c in enumerate(train_y):
        items_per_class[c].append(ii)
    for i in range(args.epochs):
        #each epoch, generate index and group into buckets, key is the index(balance dataset) is fixed 
        # this is the ind that will be used to group to form class balanced set 
        # use train_ind to do bucket grouping instead the origial index 
        # get train_ind is the key, derive from train_length and train_y
        train_ind = []
        for _ in range(len(train_y)):
            class_i = random.choices(list(range(len(class_prob))), weights=class_prob, k=1)[0]
            index = random.choice(items_per_class[class_i])
            train_ind.append(index)
        train_resample_length = [train_length[i] for i in train_ind]
        train_hist, _ = np.histogram(train_resample_length, bins=len_range)
        bucket_boundaries = generate_buckets(args.bucket_size, train_hist)
        train_pack = list(zip(train_ind, train_resample_length))
        # this is the sampler that groups the train video by their frame lengths
        train_sampler = prepare_data_divide.BySequenceLengthSampler(train_pack, bucket_boundaries, args.batch_size)
        train_set = prepare_data_divide.Dataset(args, train_list, train_y, class_prob, transform=train_transform, aug_transform = aug_transform)
        train_loader = data.DataLoader(train_set, batch_size= 1, batch_sampler=train_sampler, collate_fn=prepare_data_divide.col_fn)
        
        N_count = 0  
        epoch_loss, all_y, all_y_pred = 0, [], []
        cnn_encoder.train()
        for batch_idx, (X, X_lengths, y) in enumerate(train_loader):
             X, X_lengths, y = X.to(device), X_lengths.view(-1, ), y.to(device).view(-1, )
             N_count += X.size(0)
             model_opt.zero_grad() 
             output = cnn_encoder(X) # output is (4, 10, 5)
#             print(output.size())
             output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(X_lengths)])
        
             loss = F.cross_entropy(output_true, y)  # mini-batch loss
             epoch_loss += F.cross_entropy(output_true, y, reduction='sum').item()  # sum up mini-batch loss
             y_pred = torch.max(output_true, 1)[1]  # y_pred != output
             # collect all y and y_pred in all mini-batches 
             all_y.extend(y)
             all_y_pred.extend(y_pred) 
             # to compute accuracy
             step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy()) 
             loss.backward() 
             model_opt.step()
             # show information
             if (batch_idx + 1) % log_interval == 0:
                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                  i + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), 
                  loss.item(), 100 * step_score)) 
                 
        epoch_loss /= len(train_loader)
        # compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        epoch_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
        print('\nTrain set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), epoch_loss, 100* epoch_score))
        epoch_train_losses.append(epoch_loss)
        epoch_train_scores.append(epoch_score)
        ##start validation 
        cnn_encoder.eval()
        test_loss = 0
        all_y, all_y_pred = [], []
        with torch.no_grad():
            for X, X_lengths, y in dev_loader:
            # distribute data to device
                X, X_lengths, y = X.to(device), X_lengths.view(-1, ), y.to(device).view(-1, )

                output = cnn_encoder(X)
                output_true = torch.stack([output[i, :l, :].mean(dim=-2) for i, l in enumerate(X_lengths)])

                loss = F.cross_entropy(output_true, y, reduction='sum')
                test_loss += loss.item()                 # sum up minibatch loss
                y_pred = output_true.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

                # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(dev_loader.dataset)

        # compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
        
        epoch_dev_losses.append(test_loss)
        epoch_dev_scores.append(test_score)
        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_dev_losses)
        D = np.array(epoch_dev_scores)
        np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'training_loss.npy'), A)
        np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'training_score.npy'), B)
        np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'dev_loss.npy'), C)
        np.save(os.path.join(args.save_model_path, args.checkpoint_name, 'dev_score.npy'), D)

        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

        # save Pytorch models of best record
        if test_score > best_acc:
            best_acc = test_score 
            torch.save(cnn_encoder.state_dict(), os.path.join(args.save_model_path, args.checkpoint_name, 'cnn_encoder_acc_%.2f_epoch%d.pth'%(best_acc, i + 1)))  # save spatial_encoder
#            torch.save(model_opt.state_dict(), os.path.join(args.save_model_path, args.checkpoint_name, 'optimizer_acc_epoch{}.pth'.format(i + 1)))      # save optimizer
            print("Epoch {} model saved!".format(i + 1))
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(cnn_encoder.state_dict(), os.path.join(args.save_model_path, args.checkpoint_name, 'cnn_encoder_loss_%.2f_epoch%d.pth'%(best_loss, i + 1)))  # save spatial_encoder
#            torch.save(model_opt.state_dict(), os.path.join(args.save_model_path, args.checkpoint_name, 'optimizer_loss_epoch{}.pth'.format(i + 1)))      # save optimizer
            print("Epoch {} model saved!".format(i + 1))
            patience_log = 0 
        else:
            patience_log += 1 
        
        if patience_log >= args.patience:
            # early stopping 
            break