1. FEVAR_dataset: a folder for labeled images, pytorch dataloader loads from here
2. FEVAR_acc: save an excel file linking the original DICOM to the target folder in the FEVAR_dataset
3. Marked-cases: unsure cases saved as gif for double check
4. generate_labels.py: main script to label the orginal DICOM files
5. Prerequisite:
pydicom                   2.3.0
matplotlib                3.5.2                    
numpy                     1.22.3           
opencv-python             4.6.0.66                 
pandas                    1.4.3            
pillow                    9.2.0                    
python                    3.9.12               

6. Before running the script, the Ipython Console ---> Graphics Backend needs to be automatic not inline inorder to scroll through DICOM files 

