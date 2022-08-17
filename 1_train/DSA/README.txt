1. To perform the training, run main_dsa.py
   e.g. python main_dsa.py --encoder_model res --checkpoint_name test --patience 50 --img_size 512 --val_set 1 6 19 --resample --no_decoder --use_roi --data_aug

   will initiate a resnet18 model, save model weights in./checkpoints/test/, earlystopping with a patience of 50, 
   use index [1, 6, 19] in the dataset folder for validation, resizing images to 512 by 512, 
   use resample to balance the datatset, no RNN decoder, use ROI to get rid of empty edges, use data augmentation

   e.g. python main_dsa.py --encoder_model res_decode --checkpoint_name test  --img_size 512 --val_set 1 6 19 --patience 50 --data_aug --use_roi --resample 
    --no_encode_fc --CNN_embed_dim 512

   will initiate a resnet18 model, save model weights in./checkpoints/test/, earlystopping with a patience of 50, 
   use index [1, 6, 19] in the dataset folder for validation, resizing images to 512 by 512, 
   use resample to balance the datatset, use ROI to get rid of empty edges, use data augmentation,
   use a LSTM decoder, the resnet encoder has no fc layers before feeding into the LSTM 

   e.g. python main_dsa.py --encoder_model res_decode --checkpoint_name test  --img_size 512 --val_set 1 6 19 --patience 50 --data_aug --use_roi --resample 
    
   will initiate a resnet18 model, save model weights in./checkpoints/test/, earlystopping with a patience of 50, 
   use index [1, 6, 19] in the dataset folder for validation, resizing images to 512 by 512, 
   use resample to balance the datatset, use ROI to get rid of empty edges, use data augmentation,
   use a LSTM decoder, the resnet encoder has 2 fc layers before feeding into the LSTM 

2. To perform the validation run eval_dsa.py 
   e.g. python eval_dsa.py --encoder_model res --checkpoint_name test --weights_name acc_0.67_epoch52.pth --img_size 512 --val_set 2 8 15 --use_roi --no_decoder 
   
   will perform evaluation using a resnet18 model, loading weights from ./checkpoints/test/cnn_encoder_acc_0.67_epoch52.pth
   use index [2, 8, 15] in the dataset folder for validation, resizing images to 512 by 512, 
   no RNN decoder, use ROI to get rid of empty edges,
