#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
import sys
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import augmentation
import numpy as np

get_ipython().magic(u'matplotlib inline')

modality = ['rgb/','GT_color/'];
augmented = ['rgb_aug/','GT_color_aug/']

# RGB images
path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb/"
aug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb_aug/"

imglist=os.listdir(path)
imglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

# GT images
GTpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color/"
GTaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color_aug/"

GTimglist=os.listdir(GTpath)
GTimglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

#The sorting ensures that the corresponing images in both folders are the same. 


# Loop through all the images
for i in range(1, len(imglist)):
    
    
    # Attempt to open an image file
    rgb_filepath = path+imglist[i]
    rgb_filename = os.path.splitext(imglist[i])[0]
    rgb_image = cv2.imread(rgb_filepath)
    
    # Save the RGB image from jpg to png in the augmented images folder. Also resize them to the same size.
    
    original_image = Image.fromarray(rgb_image)
    size = (768,384)
    original_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
    #original_image.save(aug_path+rgb_filename+'.png')
    
    rgb_image = np.asarray(original_image)
    cv2.imwrite(aug_path+rgb_filename+'.png',rgb_image)


    # GT images. 
    gt_filepath = GTpath+GTimglist[i]
    gt_filename =  os.path.splitext(GTimglist[i])[0]
    gt_image = cv2.imread(gt_filepath)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        
    
    # Copy the GT image as is to the augmented images folder
    
    original_gt_image = Image.fromarray(gt_image)
    original_gt_image = ImageOps.fit(original_gt_image, size, Image.ANTIALIAS)
    #original_gt_image.save(GTaug_path+gt_filename+'.png') 
    
    gt_image = np.asarray(original_gt_image)
    cv2.imwrite(GTaug_path+gt_filename+'.png',gt_image)

    #Applying augmentations
    
    rgb_image,_ = augmentation.RANDOM_BRIGHTNESS(rgb_image, min_bright=-50, max_bright=40) 
    rgb_image, _ = augmentation.RANDOM_NOISE(rgb_image, amount=15, noise_chance=0.5)
    rgb_image, _ = augmentation.RANDOM_BLUR(rgb_image, blur_range=(0, 5), many=False)
    
    rgb_image, gt_image = augmentation.RANDOM_FLIP(rgb_image,gt_image)
    rgb_image, gt_image,_ = augmentation.RANDOM_SHIFTS(rgb_image, gt_image, h_shift=40, v_shift=70)
    rgb_image, gt_image,_ = augmentation.RANDOM_ROTATIONS(rgb_image, label=gt_image, degree=6)
    
    # Save the augmented RGB image in the augmented images folder
    cv2.imwrite(aug_path+rgb_filename+'aug.png',rgb_image)
    
    # Save the augmented GT image in the augmented images folder
    cv2.imwrite(GTaug_path+gt_filename+'aug.png',gt_image)


# In[5]:


import sys
import os
import cv2
import cv2
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import augmentation
import numpy as np

get_ipython().magic(u'matplotlib inline')

modality = ['rgb/','GT_color/'];

# RGB images
path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/rgb/"
path_aug = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/rgb_aug/"


imglist=os.listdir(path)
imglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

# GT images
GTpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/GT_color/"
GTpath_aug = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/GT_color_aug/"


GTimglist=os.listdir(GTpath)
GTimglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

#The sorting ensures that the corresponing images in both folders are the same. 


# Loop through all the images
for i in range(1, len(imglist)):
    
    
    # Attempt to open an image file
    rgb_filepath = path+imglist[i]
    rgb_filename = os.path.splitext(imglist[i])[0]
    rgb_image = cv2.imread(rgb_filepath)
    
    # Save the RGB image from jpg to png in the augmented images folder. Also resize them to the same size.
    
    original_image = Image.fromarray(rgb_image)
    size = (768,384)
    original_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
    #original_image.save(aug_path+rgb_filename+'.png')
    
    rgb_image = np.asarray(original_image)
    cv2.imwrite(path_aug+rgb_filename+'.png',rgb_image)

    # GT images. 
    gt_filepath = GTpath+GTimglist[i]
    gt_filename =  os.path.splitext(GTimglist[i])[0]
    gt_image = cv2.imread(gt_filepath)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        
    
    # Copy the GT image as is to the augmented images folder
    
    original_gt_image = Image.fromarray(gt_image)
    original_gt_image = ImageOps.fit(original_gt_image, size, Image.ANTIALIAS)
    #original_gt_image.save(GTaug_path+gt_filename+'.png') 
    
    gt_image = np.asarray(original_gt_image)
    cv2.imwrite(GTpath_aug+gt_filename+'.png',gt_image)



# In[2]:


i


# In[ ]:




