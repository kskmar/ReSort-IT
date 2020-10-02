#importing modules
import json
import cv2
import numpy as np
import os
import re
import fnmatch
from PIL import Image
# import pycococreatortools
from pycococreatortools import pycococreatortools 

ROOT_DIR = r'D:\gp_dataset\bottle1CL' #'C:/Users/anasa/Desktop/Datasets/forTrainAlu1BG'
#IMAGE_DIR = os.path.join(ROOT_DIR, "test_im")
IMAGE_DIR = os.path.join(ROOT_DIR, "train")
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "recy_annotations")
   
INFO = {"description": "Example Dataset", "url": "",  "version": "0.1.0", "year": 2020, "contributor": "cvrl", "date_created": "2020"}

LICENSES = [
    
    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License" }
]

CATEGORIES = [
        {"supercategory": "recycle","id": 3,"name": "Bottle"}, 
]

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
            if(area>1500):    #            if(area>15000):
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
    files = [f for f in files if re.match(file_types, f)]   
    return files  
        
def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 0
    # segmentation_id = 1
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        for image_filename in image_files:           
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            print (image.size)
            coco_output["images"].append(image_info)
  
            # # filter for associated png annotations
            # for root, _, files in os.walk(ANNOTATION_DIR):
            class_id = image_filename.split("_")[-2][-1]   # SOS ayto allagh !!! isxyei gia to diko mou path !
            print ("File=", image_filename)
            print (class_id)
            annotation_info = create_annotation(image_filename, image_id, class_id)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                # segmentation_id = segmentation_id + 1
            image_id = image_id + 1
            
    json_name = '1clBottleTrainBG.json'
    json_path = '{}/annot/'+ json_name
    with open(json_path.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()


          


   
   
