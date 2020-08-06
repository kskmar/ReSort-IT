# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:39:49 2020

@author: anasa
"""

#importing modules
import matplotlib.pyplot as plt
import numpy as np
import cv2



img = cv2.imread(r'D:\gp_dataset\bottle1CL\3_bottles\3_0.png')
# image_copy = np.copy(img)
# plt.imshow(img)

# image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
# plt.imshow(image_copy)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
# #definig the range of red color   
red_lower=np.array([0,0,100],np.uint8)
red_upper=np.array([250,255,255],np.uint8)  

mask = cv2.inRange(hsv, red_lower, red_upper)
print (mask)
plt.imshow(mask, cmap='gray') 

masked_image = np.copy(img)
masked_image[mask == 0] = [0, 0, 0]
plt.imshow(masked_image)


imgBack = cv2.imread(r'D:\gp_dataset\bottle1CL\bgs\rnd1.png')
#imgBack = cv2.cvtColor(imgBack, cv2.COLOR_BGR2RGB)
crop_background = imgBack[0:800, 0:800] #the image is: <class 'numpy.ndarray'>  with dimensions: (514, 816, 3)

crop_background[mask != 0] = [0,0,0]
plt.imshow(crop_background)

complete_image = masked_image + crop_background
cv2.imwrite(r'D:\gp_dataset\bottle1CL\Ztrash\xxx.jpeg', complete_image)
plt.imshow(complete_image)

