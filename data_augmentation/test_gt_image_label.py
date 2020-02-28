#!/usr/bin/env python
import sys
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import augmentation
import numpy as np

GTpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color/"
GTaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color_aug/"

GTimglist=os.listdir(GTpath)
GTimglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

for i in range(0, 230):

    # GT images. 
    gt_filepath = GTpath+GTimglist[i]
    gt_filename =  os.path.splitext(GTimglist[i])[0]
    img = cv2.imread(gt_filepath)
    size = (768,384)

    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

    img_sc = np.zeros((384,768))
        
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==None,img[:,:,1]==None),img[:,:,2]==None)] = 0
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==170,img[:,:,1]==170),img[:,:,2]==170)] = 1
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==0,img[:,:,1]==255),img[:,:,2]==0)] = 2
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==51,img[:,:,1]==102),img[:,:,2]==102)] = 3
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==0,img[:,:,1]==60),img[:,:,2]==0)] = 3
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==255,img[:,:,1]==120),img[:,:,2]==0)] = 4
    img_sc[np.logical_and(np.logical_and(img[:,:,0]==0,img[:,:,1]==0),img[:,:,2]==0)] = 5
    
    # Copy the GT image to the new folder
    cv2.imwrite(GTaug_path+gt_filename+'.png',img_sc)