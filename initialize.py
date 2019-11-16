#  from __future__ import print_function
# import zipfile
import os
import cv2
# import torchvision.transforms as transforms
import numpy as np
# from skimage import exposure
# import torch
# import PIL

 
def initialize_data(folder):
    train_folder = folder + '/train'
    val_folder = folder + '/val'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        folders = os.listdir(train_folder)
        folders = [folder for folder in folders if folder[0]!='.']
        for dirs in folders:
            files = os.listdir(train_folder + '/' + dirs)
            files = [file for file in files if file[0]!="."]
            random_files = np.random.choice(files,int(0.2*len(files)),replace=False)
            if not os.path.isdir(val_folder + '/' + dirs):
                os.mkdir(val_folder + '/' + dirs)
            
            #if dirs.startswith('000'):
             #   os.mkdir(val_folder + '/' + dirs)
            for f in random_files:
                print(train_folder + '/' + dirs + '/' + f,val_folder + '/' + dirs + '/' + f)  
              #if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
        print("Done Making Validation Set")
        return
    print("Validation Set exists!")