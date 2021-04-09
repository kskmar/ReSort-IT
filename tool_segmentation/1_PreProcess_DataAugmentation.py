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

def renameFigs(image_paths, dir_name):
    i = 10000
    for image_path in image_paths:
        img=cv2.imread(image_path)
        height, width = img.shape[:2]
        if height!=800 or width!=800:
            img = cv2.resize(img, (800,800))
        cv2.imwrite(dir_name + str(i)+".jpg", img)
        i -= 1 

def replaceBG(img, imgBack, imgname):
   
    # hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    # # plt.imshow(hsv) 
    # # plt.show()
    # # #definig the range of red color   
    # red_lower=np.array([0,0,100],np.uint8)
    # red_upper=np.array([250,255,255],np.uint8)  
    
    # mask = cv2.inRange(hsv, red_lower, red_upper)
    
    # result = cv2.bitwise_and(img, img, mask=mask)
    # b, g, r = cv2.split(result)  
    # filter = g.copy()    
    # ret,mask = cv2.threshold(filter,0,255, 1)
    # img[ mask == 0] = 255
    # plt.imshow(img)
    # plt.show()
    
    
    complete_image = img + imgBack
    # class_id = 3  #### SOOOOS change it ! #######
    # cv2.imwrite('/media/microralp/MKOSK/instrument_seg/test/'+ "_" + imgname +'.png', complete_image)
    # cv2.imwrite('/media/microralp/MKOSK/instrument_seg/orange_1_img/imgs/new2/' + 'segimg' + str(image_id) +'.png', img) 
    #plt.imshow(complete_image)


def searchD(f_str, image_GT):
    numGT = 0
    k = len(f_str)
    for i in range(len(image_GT)):
        if f_str in str(image_GT[i].split("/")[-1]):
            numGT = i
            # print ( numGT, f_str, gt_name[numGT])
            return numGT
  

############### MAIN ######################

# mode = ["train", "val"]

image_id = 124

input_dir = '/media/microralp/MKOSK/instrument_seg/orange_1_img/img_left' 
image_paths = []
for filename in os.listdir(input_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(input_dir, filename))
        
bg_dir = '/media/microralp/MKOSK/instrument_seg/liver_1_img/img_left' 
bg_paths = []
for fn in os.listdir(bg_dir):
    if os.path.splitext(fn)[1].lower() in ['.png', '.jpg', '.jpeg']:
        bg_paths.append(os.path.join(bg_dir, fn))
        
mask_dir = '/media/microralp/MKOSK/instrument_seg/orange_1_img/img_synthetic_new_masksRR' 
masks_paths = []
for filename in os.listdir(mask_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        masks_paths.append(os.path.join(mask_dir, filename))
    
for n in range(len(image_paths)):
    print(image_paths[n])
    image_name = image_paths[n].split("/")[-1]
    f_str = image_paths[n].split("/")[-1].split(".")[0]
    image = cv2.imread(image_paths[n])    
    numGT = searchD(f_str, masks_paths)
    msk = cv2.imread(masks_paths[numGT])
    print (masks_paths[numGT])
    
    msk = cv2.bitwise_not(msk)  


    tool_mask = cv2.add(image, msk)
    img2gray = cv2.cvtColor(tool_mask,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    print (tool_mask.shape)

    # plt.imshow(mask)     
    # plt.show()  


    cv2.imwrite('/media/microralp/MKOSK/instrument_seg/orange_1_img/img_synthetic_new_masks_XX/' +  f_str + "XXmask" +'.png', mask)  
    bgchoice = random.choice(bg_paths)
    imgBack = cv2.imread(bgchoice)
    imgBack
    # img2 = cv2.bitwise_and(imgBack, tool_mask)   
    img2 = cv2.bitwise_and(image, image, mask=mask) 
    
    # cv2.imwrite('/media/microralp/MKOSK/instrument_seg/orange_1_img/new/' +  f_str + str(image_id) +'X.png', img2)   
    back_croped = cv2.bitwise_and(imgBack, imgBack, mask=mask_inv)     
    img3 = cv2.add(img2, back_croped) #cv2.bitwise_and(imgBack, img2)     #img2 = cv2.bitwise_and(imgBack, msk)
    cv2.imwrite('/media/microralp/MKOSK/instrument_seg/orange_1_img/img_synthetic_new_XX/' +  f_str + 'XX.png', img3) 

    # image_id = image_id + 1

print(image_id)
