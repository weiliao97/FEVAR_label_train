1. Data prerequisite:
  1a. Folders containing the validation set data, such as BIDMC-case2, HamburgUKE-case3, Utrecht-UMCU-case4
      Also need another folder DSA with the DSA validaiton sets inside
  1b. csv files generated while labeling the data, will use the timestamps from it 
      Also need all the dsa csv files linking the original timestamps
  1c. model weights 

2. Main file is DSA_timeline.py


3. Prerequisites:
   customtkinter 4.5.5
   torch         1.12.0
   torchvision   0.13.0
   scikit-learn  1.1.1                    
   scipy         1.8.1
   python        3.9.12
   pandas        1.4.3
   pillow        9.2.0
   numpy         1.22.3
   opencv-python 4.6.0.66