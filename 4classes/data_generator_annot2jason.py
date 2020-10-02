# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:39:49 2020

@author: anasa
"""

#importing modules

import random

#importing modules
import json
import cv2
import numpy as np
import os
import re
import fnmatch
from pycococreatortools import pycococreatortools 
import sys

ROOT_DIR = 'D:/Datasets/big_dataset/'    
IMAGE_DIR = os.path.join(ROOT_DIR, "val")

from datetime import datetime
now = datetime.now()
current_time = now.strftime(("%Y%m%dT%H%M"))
print("Current Time =", current_time)
current_filename = os.path.basename(sys.argv[0]).split('/')
print (current_filename[-1])


INFO = {"description": "Example Dataset", "url": "",  "version": "0.1.0", "year": 2020, "contributor": "cvrl", "date_created": "2020"}

LICENSES = [
    
    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License" }
]

CATEGORIES = [
        {"supercategory": "recycle","id": 1,"name": "Alu"}, 
        {"supercategory": "recycle","id": 2,"name": "Carton"},
        {"supercategory": "recycle","id": 3,"name": "Bottle"},
        {"supercategory": "recycle","id": 4,"name": "Nylon"},
]

def replaceBG(img, imgBack):  #, imgname
    
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    # #definig the range of red color   
    red_lower=np.array([0,0,100],np.uint8)
    red_upper=np.array([250,255,255],np.uint8)  
    
    mask = cv2.inRange(hsv, red_lower, red_upper)
    
    masked_image = np.copy(img)
    masked_image[mask == 0] = [0, 0, 0]
    #plt.imshow(masked_image) 
   
    #imgBack = cv2.cvtColor(imgBack, cv2.COLOR_BGR2RGB)
    crop_background = imgBack[0:800, 0:800] #the image is: <class 'numpy.ndarray'>  with dimensions: (514, 816, 3)
    
    crop_background[mask != 0] = [0,0,0]
    # plt.imshow(crop_background)
    
    complete_image = masked_image + crop_background
    return complete_image
    #cv2.imwrite('C:/Users/anasa/Desktop/Datasets/forTrainAlu1BG/valbg/'+ "1_" + imgname, complete_image)
    #plt.imshow(complete_image)


############### TEST ######################
def ret_path(img_dir):
    img_paths = []
    for filename in os.listdir(img_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            img_paths.append(os.path.join(img_dir, filename))
    return img_paths

  
def create_annotation(im_path, image_id, category_id):
    IMG=cv2.imread(im_path)
    img=IMG    
    #converting frame(img i.e BGR) to HSV (hue-saturation-value)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    #definig the range of red color   
    red_lower=np.array([0,0,100],np.uint8)
    red_upper=np.array([250,255,255],np.uint8)   
     #finding the range of red,blue and yellow color in the image
    red=cv2.inRange(hsv, red_lower, red_upper)
    #Morphological transformation, Dilation
    kernal = np.ones((5 ,5), "uint8")    
    red=cv2.dilate(red,kernal)
    res=cv2.bitwise_and(img, img, mask = red)    
    blobsINFO=np.zeros((10, 3))   
    contours,_=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite(path_folder +"/recy_images/"+str(category_id)+"_"+str(num_id)+".png", img)
    i = 1
    for pic, contour in enumerate(contours):
            segmentation = []            
            area = cv2.contourArea(contour)
            # print ("AREA", area)
            if(area>400):    #            if(area>15000):
                areaMAX=area
                x,y,w,h = cv2.boundingRect(contour)
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)             
                blobsINFO[i,0]=(-1.2422*((x+w/2)-320))
                blobsINFO[i,1]=(1.2422*((y+h/2)-240))-993   #-982
                blobsINFO[i,2]=1
                cv2.putText(img,"RED",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                    #blobsINFO[i,1]=
                    #blobsINFO[i]=s
                i=i+1
                ellipse=cv2.fitEllipse(contour)
                cv2.ellipse(img,ellipse,(0,255,0),2)
                # print(ellipse)
                Exy=ellipse[0]
                EMAma=ellipse[1]
                angle=ellipse[2]
                angle2=180-angle
                offset_elips=EMAma[1]*0.25
                Xpick=np.sin(angle2*0.0174532925)*offset_elips
                Ypick=np.cos(angle2*0.0174532925)*offset_elips
                Xpix=Exy[0]+Xpick
                Ypix=Exy[1]+Ypick
                cv2.putText(img,".Exy",(int(Exy[0]),int(Exy[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255,5))
                cv2.putText(img,".",(int(Xpix),int(Ypix)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),5)
                c_img = cv2.drawContours(img, contour, -1, (0, 255, 0), 1)            
                contour = contour.flatten().tolist()
                if len(contour) > 4:
                    segmentation.append(contour)
                if len(segmentation) == 0:
                        continue
                #print (category_id)
                annotation = {'image_id': image_id,
                              'category_id': category_id,
                              'iscrowd': 0,
                              'segmentation': segmentation,                             
                              'area': area,
                              'bbox': [x,y,w,h],
                              }                
    return annotation
    
                    
def filter_for_jpeg(root, files):
    file_types = ['*.png', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types(), f)]   
    return files  


def mainfun(f_path,f2_path, bg_path, k, image_id, rangeN, class_idX, class_idY, coco_output, imageDir = IMAGE_DIR):
    for i in range(rangeN):   
        image_filename = str(k) + ".jpg"
        img = cv2.imread(random.choice(f_path))

        print (img.shape[:2])
        image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), img.shape[:2])
        coco_output["images"].append(image_info)
        class_id =  class_idX                    #image_filename.split("_")[-2][-1]   # SOS ayto allagh !!! isxyei gia to diko mou path !
        print ("File=", image_filename)
        print ("class_id", class_id)

        im_path= IMAGE_DIR + "/"+  image_filename
        cv2.imwrite(im_path, img)
        annotation_info = create_annotation(im_path, image_id, class_id)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
            # segmentation_id = segmentation_id + 1
        bgname = random.choice(bg_path)
        imgBack = cv2.imread(bgname)
        imFirst = replaceBG(img, imgBack)
        
        if f2_path!=None:
            img2 = cv2.imread(random.choice(f2_path))
            cv2.imwrite(im_path, img2)     
            class_id =  class_idY                    #image_filename.split("_")[-2][-1]   # SOS this works for this  !
            print ("File=", image_filename)
            print (class_id)
            annotation_info = create_annotation(im_path, image_id, class_id)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            
            imFinal = replaceBG(img2, imFirst)
            cv2.imwrite(im_path, imFinal)
            image_id = image_id + 1   
            k -=1
        else:
            cv2.imwrite(im_path, imFirst)
            image_id = image_id + 1   
            k -=1           
    return coco_output, k, image_id
        
def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    p = 0
    image_id = 0
    segmentation_id = 1
    k = 20000
    # filter for jpeg images
    back_dir = 'D:/Datasets/bigDataset4cl/train/bg/'
    bg_path = ret_path(back_dir)
    rangeN=350
    
    data_dir = 'D:/Datasets/bigDataset4cl/train/'
    dic_data= {1:data_dir+'cl_1/', 2:data_dir+'cl_2/', 
               3:data_dir+'cl_3/', 4:data_dir+'cl_4/'}

# For n number of classes     ====> n =4 classes
    for class_idX in dic_data:
            f_dir = dic_data[class_idX]
            f_path = ret_path(f_dir)
            coco_output, k, image_id =  mainfun(f_path, f_path, bg_path, k, image_id, rangeN, class_idX, class_idX, coco_output)

# For number of pairs  nx(n-1) ===> 12 times for n = 4 classes
    for class_idX in dic_data:
        f_dir = dic_data[class_idX]
        f_path = ret_path(f_dir)
        for class_idY in dic_data:
            if class_idX!=class_idY:
                p +=1
                f2_dir =  dic_data[class_idY]
                f2_path = ret_path(f2_dir)
                coco_output, k, image_id =  mainfun(f_path, f2_path, bg_path, k, image_id, rangeN, class_idX, class_idY, coco_output)
                print ("Times ==> ", p, class_idX, class_idY )   
                 
      
    json_name ='big_dataset_val_V1' + '.json'
    json_path = ROOT_DIR + 'annot/'+ json_name
    with open(json_path.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()


