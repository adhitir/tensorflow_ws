#!/usr/bin/env python
#test

import os


path_rgb = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/rgb"
path_depth = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/depth_color"
path_evi = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/evi_color"

path_label = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/GT_index"

imglist=os.listdir(path_depth)
imglist.sort(key=lambda f: int(filter(str.isdigit, f))) #sorting the images by number

path_rgb_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/rgb_aug"
path_depth_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/depth_color"
path_evi_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/evi_color"

path_label_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/GT_color_aug"

with open('test.txt', 'wa') as f:
    for i in range(len(imglist)):
        imgname = os.path.splitext(imglist[i])[0]
        #print >> f,"%s/%s %s/%s" % (path_rgb_cluster,imgname+'_Clipped.png',path_label_cluster,imgname+'_Clipped.png')
        print >> f,"%s/%s %s/%s" % (path_depth_cluster,imgname+'.png',path_label_cluster,imgname+'_Clipped.png')
        #print >> f,"%s/%s %s/%s" % (path_evi_cluster,imgname+'.png',path_label_cluster,imgname+'_Clipped.png')

        #print >> f,"%s/%s %s/%s" % (path_rgb,imgname+'_Clipped.png',path_label,imgname+'_Clipped.png')
