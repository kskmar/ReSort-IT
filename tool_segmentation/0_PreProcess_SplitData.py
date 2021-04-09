#importing modules

import cv2
import numpy as np
import os
import re
import fnmatch


# ROOT_DIR = '/home/microralp/mk_dev/robot-surgery-segmentation/data/AnnotatedImages/'   

ROOT_DIR = '/home/microralp/mk_dev/robot-surgery-segmentation/data/Segmentation_Rigid_Training/Training/OP4'
#IMAGE_DIR = os.path.join(ROOT_DIR, "test_im")
IMAGE_DIR = '/media/microralp/MKOSK/instrument_seg/orange_1_img/img_left/'
mask_DIR = '/media/microralp/MKOSK/instrument_seg/orange_1_img/mask/'  
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "recy_annotations")
   

                    
def filter_for_jpeg(root, files):
    file_types = ['*.png', '*.jpg', '*.bmp']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]   
    # print ("here", files)
    return files  

def searchD(f_str, image_GT):
    numGT = 0
    k = len(f_str)
    for i in range(len(image_GT)):
        gt_name = image_GT[i].split("/")[-1]
        # print (f_str == str(image_GT[i].split("/")[-1])[0:k-1])
        # print (f_str)
        # print (str(image_GT[i].split("/")[-1])[:k-1])
        if f_str in str(image_GT[i].split("/")[-1]):
            numGT = i
            # print ( numGT, f_str, gt_name[numGT])
            return numGT
  
  
IMAGE_Dest = '/home/microralp/mk_dev/robot-surgery-segmentation/data/Tracking_Rigid_Training/raw_all/'   
mask_Dest = '/home/microralp/mk_dev/robot-surgery-segmentation/data/Tracking_Rigid_Training/mask_all/'   

IMAGE_Dest =  '/media/microralp/MKOSK/instrument_seg/orange_1_img/img_left/'
mask_Dest = '/media/microralp/MKOSK/instrument_seg/orange_1_img/mask/'  

image_files = []   
image_GT= []    
image_id = 300

for root, _, files in os.walk(IMAGE_DIR):
    image_files = filter_for_jpeg(root, files)
    
for root, _, files in os.walk(mask_DIR):
    image_GT = filter_for_jpeg(root, files)
    
    
for n in range(len(image_GT)):
    print(image_GT[n])
    # if "instrument" in image_GT[n].split("/")[-1]:
        # os.remove(os.path.join(image_GT[n]))

    image_name = image_GT[n].split("/")[-1]
    f_str = image_GT[n].split("/")[-1].split("class")[0]
    numGT = searchD(f_str, image_files)
    print (IMAGE_Dest + "img_" + str(image_id) +'.png')
    image = cv2.imread(image_files[numGT])
    msk = cv2.imread(image_GT[n])
    cv2.imwrite(IMAGE_Dest + "segimg" + str(image_id) +'.png', image)  
    cv2.imwrite(mask_Dest + "segimg" + str(image_id) +'mask.png', msk)   
    image_id = image_id + 1

 
        
            




          


   
   
