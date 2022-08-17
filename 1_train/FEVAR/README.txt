1. To perform the training, run main_divide.py
   e.g. python main_divide.py --encoder_model res --checkpoint_name test --patience 50 --img_size 512 --val_set 1 4 15 --resample --no_decoder --use_roi --data_aug

   will initiate a resnet18 model, save model weights in./checkpoints/test/, earlystopping with a patience of 50, 
   use index [1, 4, 15] in the dataset folder for validation, resizing images to 512 by 512, 
   use resample to balance the datatset, no RNN decoder, use ROI to get rid of empty edges, use data augmentation
   
2. To perform the validation run eval_divide.py 
   e.g.  python eval_divide.py --encoder_model res --checkpoint_name test --weights_name acc_0.85_epoch86.pth --val_set 1 4 15 --img_size 512 --use_roi --no_decoder --resample 

   will perform evaluation using a resnet18 model, loading weights from ./checkpoints/test/cnn_encoder_acc_0.85_epoch86.pth
   use index [1, 4, 15] in the dataset folder for validation, resizing images to 512 by 512, 
   no RNN decoder, use ROI to get rid of empty edges,

3. Prerequisites:
numpy                     1.22.3         
opencv-python             4.6.0.66                
pandas                    1.4.2           
pillow                    9.0.1           
python                    3.10.4               
pytorch                   1.11.0          
scikit-learn              1.1.1           
scipy                     1.8.1           
torchvision               0.12.0             


   
