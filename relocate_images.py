# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:00:10 2020

@author: maria
"""

import os
import sys
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage
import cv2

# ROOT_DIR = 'F:/Fredy/Dataset/data/train/Carton'
# assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'


CATEGORIES = [
        {"supercategory": "recycle","id": 1,"name": "Alu"}, 
        {"supercategory": "recycle","id": 2,"name": "Carton"},
        {"supercategory": "recycle","id": 3,"name": "Bottle"},
        {"supercategory": "recycle","id": 4,"name": "Nylon"},
]
############### TEST ######################

input_dir ='D:/Datasets/bigDataset4cl/train/bg'
image_paths = []
for filename in os.listdir(input_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(input_dir, filename))

i = 10000
for image_path in image_paths:
    img=cv2.imread(image_path)
    height, width = img.shape[:2]
    if height!=800 or width!=800:
        img = cv2.resize(img, (800,800))
    #print (img)
#    cv2.imwrite(path_folder +"/recy_images/"+str(category_id)+"_"+str(num_id)+".png", img)
    cv2.imwrite("D:/Datasets/bigDataset4cl/train/bg/bb/"+ str(i)+".jpg", img)
    i -= 1 
