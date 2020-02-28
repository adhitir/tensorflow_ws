#!/usr/bin/env python
import sys
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import augmentation
import numpy as np

#get_ipython().magic(u'matplotlib inline')

#modality = ['rgb/','GT_color/'];
#augmented = ['rgb_aug/','GT_color_aug/']

# RGB images
RGBpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb/"
DEPTHpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/depth_color/"
EVIpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/evi_color/"

RGBaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb_aug/"
DEPTHaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/depth_aug/"
EVIaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/evi_aug/"

# RGB image sorting
imglist=os.listdir(DEPTHpath)
imglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

# GT images
GTpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_index/"
GTaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_index_aug/"

GTimglist=os.listdir(GTpath)
GTimglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

#The sorting ensures that the corresponing images in both folders are the same. 

# for i in range(0, len(imglist)):
#     rgb_filepath = RGBpath+imglist[i]
#     rgb_filename = os.path.splitext(imglist[i])[0]
#     rgb_image = cv2.imread(rgb_filepath)
    
#     original_image = Image.fromarray(rgb_image)
#     #size = (768,384)
#     #original_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
    
#     rgb_image = np.asarray(original_image)
#     cv2.imwrite(RGBpathPNG+rgb_filename+'.png',rgb_image)
# GTpath_color = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color/"

# for i in range(0, len(GTimglist)):

#     # GT images. 
#     gt_filepath = GTpath+GTimglist[i]
#     gt_filename =  os.path.splitext(GTimglist[i])[0]
#     gt_image = cv2.imread(gt_filepath)
   
#     size = (768,384)
#     gt_image = cv2.resize(gt_image, dsize=size, interpolation=cv2.INTER_CUBIC)
    
#     cv2.imwrite(GTpath+gt_filename+'.png',gt_image)


# Loop through all the images
for i in range(0, len(imglist)): 
        
    # Open and read images
    rgb_filepath = RGBpath+imglist[i]
    depth_filepath = DEPTHpath+imglist[i]
    evi_filepath = EVIpath+imglist[i]
    gt_filepath = GTpath+GTimglist[i]


    rgb_image = cv2.imread(rgb_filepath)
    depth_image = cv2.imread(depth_filepath)
    evi_image = cv2.imread(evi_filepath)
    gt_image = cv2.imread(gt_filepath)


    # Extract image name
    imgname = os.path.splitext(imglist[i])[0]

    # Resize images
    size = (768,384)
    rgb_image = cv2.resize(rgb_image, dsize=size, interpolation=cv2.INTER_CUBIC)
    depth_image = cv2.resize(depth_image, dsize=size, interpolation=cv2.INTER_CUBIC)
    evi_image = cv2.resize(evi_image, dsize=size, interpolation=cv2.INTER_CUBIC)

    # Rewrite the resized images into their own paths
    cv2.imwrite(RGBpath+imgname+'.png',rgb_image)
    cv2.imwrite(DEPTHpath+imgname+'.png',depth_image)
    cv2.imwrite(EVIpath+imgname+'.png',evi_image)
    # GT_index is already the correct size.

 
    #Applying augmentations

    #rgb_image,_ = augmentation.RANDOM_BRIGHTNESS(rgb_image, min_bright=-50, max_bright=40) 
    #rgb_image, _ = augmentation.RANDOM_NOISE(rgb_image, amount=15, noise_chance=0.5)
    #rgb_image, _ = augmentation.RANDOM_BLUR(rgb_image, blur_range=(0, 5), many=False)

    rgb_image, depth_image, evi_image, gt_image = augmentation.RANDOM_FLIP(rgb_image, depth_image, evi_image, gt_image)
    rgb_image, depth_image, evi_image, gt_image,_ = augmentation.RANDOM_ZOOMS(rgb_image, depth_image, evi_image, label=gt_image)
    #rgb_image, depth_image, evi_image, gt_image,_ = augmentation.RANDOM_ROTATIONS(rgb_image, depth_image, evi_image, label=gt_image, degree=6)

    # Save the augmented images in the augmented images folder
    cv2.imwrite(RGBaug_path + imgname+'.png',rgb_image)
    cv2.imwrite(DEPTHaug_path + imgname+'.png',depth_image)
    cv2.imwrite(EVIaug_path + imgname+'.png',evi_image)

    # Save the augmented GT image in the augmented images folder
    cv2.imwrite(GTaug_path + imgname+'_mask.png',gt_image)

