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
import matplotlib.pyplot as plt
import random

# ROOT_DIR = 'F:/Fredy/Dataset/data/train/Carton'
# assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'
#img = cv2.imread('C:/Users/anasa/Desktop/mar_test/1_0.png')

def replaceBG(img, imgBack, imgname):
    #image_copy = np.copy(img)
    #plt.imshow(img)
    
    #image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    #plt.imshow(image_copy)
    
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    # #definig the range of red color   
    red_lower=np.array([0,0,100],np.uint8)
    red_upper=np.array([250,255,255],np.uint8)  
    
    mask = cv2.inRange(hsv, red_lower, red_upper)
    #print (mask)
    #plt.imshow(mask, cmap='gray') 
    
    masked_image = np.copy(img)
    masked_image[mask == 0] = [0, 0, 0]
    #plt.imshow(masked_image)
    
    
    # imgBack = cv2.imread('C:/Users/anasa/Desktop/mar_test/BG.png')
    #imgBack = cv2.cvtColor(imgBack, cv2.COLOR_BGR2RGB)
    crop_background = imgBack[0:800, 0:800] #the image is: <class 'numpy.ndarray'>  with dimensions: (514, 816, 3)
    
    crop_background[mask != 0] = [0,0,0]
    plt.imshow(crop_background)
    
    complete_image = masked_image + crop_background
    class_id = 3  #### SOOOOS change it ! #######
    cv2.imwrite('D:/gp_dataset/bottle1CL/valbg/'+ str(class_id)+"_" + imgname, complete_image)
    #plt.imshow(complete_image)

# CATEGORIES = [
#         {"supercategory": "recycle","id": 1,"name": "Alu"}, 
#         {"supercategory": "recycle","id": 2,"name": "NonAlu"},
# ]

############### TEST ######################

input_dir = 'D:/gp_dataset/bottle1CL/val' #'C:/Users/anasa/Desktop/Datasets/multiclassNonAlu'
image_paths = []
for filename in os.listdir(input_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(input_dir, filename))
        
bg_dir = 'D:/gp_dataset/bottle1CL/bgs' #'C:/Users/anasa/Desktop/Datasets/multiclassNonAlu'
bg_paths = []
for fn in os.listdir(bg_dir):
    if os.path.splitext(fn)[1].lower() in ['.png', '.jpg', '.jpeg']:
        bg_paths.append(os.path.join(bg_dir, fn))
        
for i in range(len(image_paths)):
    imgname = image_paths[i].split("_")[-1]
    #print (imgname)                                   
    img = cv2.imread(image_paths[i])
    bgchoice = random.choice(bg_paths)
    imgBack = cv2.imread(bgchoice)
    replaceBG(img, imgBack, imgname)

    # i = 130
    # for image_path in image_paths:
    #     img=Image.open(image_path)
    #     #print (img)
    # #    cv2.imwrite(path_folder +"/recy_images/"+str(category_id)+"_"+str(num_id)+".png", img)
    #     im_crop = img.crop((1081,393, 1081+800,393+800))       #(700, 1, 1500, 801) , 635,403, 635+800,403+800)
    #     im_crop.save("C:/Users/anasa/Desktop/Datasets/forTrainAlu1BG/bgs/"+"bgr"+str(i)+".png", quality=95)
        
    #     #cv2.imwrite("C:/Users/anasa/Desktop/Datasets/forTrainAlu1BG/bgs/"+"bgr"+str(i)+".png", img)
    #     i += 1