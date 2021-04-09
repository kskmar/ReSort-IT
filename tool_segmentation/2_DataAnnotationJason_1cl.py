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
import matplotlib.pyplot as plt

ROOT_DIR = '/home/microralp/mk_dev/robot-surgery-segmentation/data/maskrcnn_datasetC/'   
#IMAGE_DIR = os.path.join(ROOT_DIR, "test_im")
IMAGE_DIR = os.path.join(ROOT_DIR, "train")
mask_DIR = os.path.join(ROOT_DIR, "masks")
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "recy_annotations")
   
INFO = {"description": "Example Dataset", "url": "",  "version": "0.1.0", "year": 2021, "contributor": "advr", "date_created": "2021"}

LICENSES = [
    
    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License" }
]

CATEGORIES = [
        {"supercategory": "surgery","id": 1,"name": "Tool"}, 
]

def create_annotation(im_path, image_id, category_id):
    IMG=cv2.imread(im_path)
    # plt.imshow(IMG)
    # plt.show()
    img=IMG    
    #converting frame(img i.e BGR) to HSV (hue-saturation-value)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    #definig the range of red color   
    red_lower=np.array([0,0,70],np.uint8)
    red_upper=np.array([250,255,255],np.uint8)   
     #finding the range of red,blue and yellow color in the image
    red=cv2.inRange(hsv, red_lower, red_upper)
    # displaying image
    # plt.imshow(red)
    # cv2.waitKey(0)
    # plt.show()
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
            if(area>500):    #            if(area>15000):
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
    for root, _, files in os.walk(mask_DIR):
        image_GT = filter_for_jpeg(root, files)
        # for image_filename in image_files:
    for num in range(len(image_files)):
        image = cv2.imread(image_files[num])
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_files[num]), image.shape)
        print (image.shape)
        coco_output["images"].append(image_info)
        # image_name = image_files[num].split("/")[-1]
        f_str = image_files[num].split("/")[-1].split(".")[0]

        numGT = searchD(f_str, image_GT)
        print ( "PAIRS-->", f_str)         
        print (image_GT[numGT])
        # # filter for associated png annotations
        # for root, _, files in os.walk(ANNOTATION_DIR):
        class_id = 1   # SOS ayto allagh !!! isxyei gia to diko mou path !
        # print ("File=", image_name)
        # print (class_id)
        # print (ROOT_DIR+'/masks/'+image_files[num].split("/")[-1])
        # if "GT" in image_files[num].split("/")[-1]:
        #     # print(image)
        #     cv2.imwrite(ROOT_DIR+'/masks/'+image_files[num].split("/")[-1], image)
        #     os.remove(os.path.join(image_files[num]))
 
        annotation_info = create_annotation(image_GT[numGT], image_id, class_id)
        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)
            # segmentation_id = segmentation_id + 1
        image_id = image_id + 1
            
    json_name = 'trainImgFulldataC.json'
    json_path = '{}/annot/'+ json_name
    with open(json_path.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()


          


   
   
