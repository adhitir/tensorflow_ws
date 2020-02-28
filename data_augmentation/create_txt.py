#!/usr/bin/env python
# coding: utf-8

# In[7]:


#train
import os

path_rgb = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/rgb_aug"
rgblist=os.listdir(path_rgb)
rgblist.sort(key=lambda f: int(filter(str.isdigit, f)))

#print(rgblist)


# In[8]:


path_label = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/train/GT_color_aug"
labellist=os.listdir(path_label)
labellist.sort(key=lambda f: int(filter(str.isdigit, f)))
#print(labellist)


# In[9]:


path_rgb_cluster = "/home/adhitir/AdapNet/1freiburg_forest/train/rgb_aug"
path_label_cluster = "/home/adhitir/AdapNet/1freiburg_forest/train/GT_color_aug"
with open('train.txt', 'a') as f:
    for i in range(len(labellist)):
        print >> f,"%s/%s %s/%s" % (path_rgb_cluster,rgblist[i],path_label_cluster,labellist[i])


# In[13]:


#test
import os

path_rgb = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/rgb_aug"
rgblist=os.listdir(path_rgb)
rgblist.sort(key=lambda f: int(filter(str.isdigit, f)))

#print(rgblist)


# In[14]:


path_label = "/home/adhitir/Semantic_Segmentation/AdapNet/1freiburg_forest/test/GT_color_aug"
labellist=os.listdir(path_label)
labellist.sort(key=lambda f: int(filter(str.isdigit, f)))
#print(labellist)


# In[15]:


path_rgb_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/rgb_aug"
path_label_cluster = "/hdd/adhitir/AdapNet/1freiburg_forest/test/GT_color_aug"
with open('test.txt', 'a') as f:
    for i in range(len(labellist)):
        print >> f,"%s/%s %s/%s" % (path_rgb_cluster,rgblist[i],path_label_cluster,labellist[i])


# In[ ]:




