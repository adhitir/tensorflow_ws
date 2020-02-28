#!/usr/bin/env python

import os

RGBpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb/"
DEPTHpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/depth_color/"
EVIpath = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/evi_color/"

RGBaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb_aug/"
DEPTHaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/depth_aug/"
EVIaug_path = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/evi_aug/"

imglist=os.listdir(DEPTHaug_path)
imglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

path_label = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_index_aug"


#labellist=os.listdir(path_label)
#labellist.sort(key=lambda f: int(filter(str.isdigit, f)))

# path in cluster

path_rgb_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/train/rgb"
path_depth_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/train/depth_color"
path_evi_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/train/evi_color"

path_rgb_cluster_aug = "/hdd/adhitir/AdapNet/1freiburg_forest/train/rgb_aug"
path_depth_cluster_aug = "/hdd/adhitir/AdapNet/1freiburg_forest/train/depth_aug"
path_evi_cluster_aug = "/hdd/adhitir/AdapNet/1freiburg_forest/train/evi_aug"

path_label_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/train/GT_index"
path_label_cluster_aug = "/hdd/adhitir/AdapNet/1freiburg_forest/train/GT_index_aug"


with open('train.txt', 'wa') as f:
    for i in range(len(imglist)):
        imgname = os.path.splitext(imglist[i])[0]
        print >> f,"%s/%s %s/%s" % (path_rgb_cluster,imgname+'.png',path_label_cluster,imgname+'_mask.png')
        print >> f,"%s/%s %s/%s" % (path_depth_cluster,imgname+'.png',path_label_cluster,imgname+'_mask.png')
        print >> f,"%s/%s %s/%s" % (path_evi_cluster,imgname+'.png',path_label_cluster,imgname+'_mask.png')

        print >> f,"%s/%s %s/%s" % (path_rgb_cluster_aug,imgname+'.png',path_label_cluster_aug,imgname+'_mask.png')
        print >> f,"%s/%s %s/%s" % (path_depth_cluster_aug,imgname+'.png',path_label_cluster_aug,imgname+'_mask.png')
        print >> f,"%s/%s %s/%s" % (path_evi_cluster_aug,imgname+'.png',path_label_cluster_aug,imgname+'_mask.png')



