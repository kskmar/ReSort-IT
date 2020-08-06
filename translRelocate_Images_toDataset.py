# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:54:02 2020

@author: anasa
"""


#importing modules
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

def download_calibration_file(serial_number) :
    if os.name == 'nt' :
        hidden_path = 'C:/Users/anasa/Desktop/ZED_Detection/'#'D:/Downloads/'
        calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'
    else :
        hidden_path = 'C:/Users/anasa/Desktop/ZED_Detection/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url+str(serial_number), out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""
    return calibration_file

def init_calibration(calibration_file, image_size) :
    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])
    config = configparser.ConfigParser()
    config.read(calibration_file)
    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False
    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])
    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)
    
    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])
    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])
    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])
    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])
    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])
    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)
    cameraMatrix_left = P1
    cameraMatrix_right = P2
    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution:
    width = 1920
    height =1080


cap = cv2.VideoCapture(0)


if cap.isOpened() == 0:
    exit(-1)

image_size = Resolution()
image_size.width = 1920
image_size.height = 1080

# Set the video resolution to HD720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
serial_number = 12970 # int(sys.argv[1])
calibration_file = download_calibration_file(serial_number)
if calibration_file  == "":
    exit(1)
print("Calibration file found. Loading...")

camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)

red_lower=np.array([0,0,75],np.uint8)


def translateImg(fimage, fname):
    image = cv2.imread(fimage)
    height, width = image.shape[:2]       
    offset_height, offset_width = random.randint(-100,200),random.randint(-100,200)  #random.randint(-(height-200)/2,(height-200)/2), random.randint(-(width-200)/2,(width-200)/2)      
    T = np.float32([[1, 0, offset_width], [0, 1, offset_height]])      
    image = cv2.resize(image, (int(800*0.7),int(800*0.7)))

    img = cv2.warpAffine(image, T, (width, height))   # We use warpAffine to transform 
      
    
    # complete_image = masked_image + crop_background
    cv2.imwrite('D:/test_images/' +"tstIm_0" +str(fname) + ".jpg", img)
    #plt.imshow(complete_image)
    
    
def ret_path(img_dir):
    img_paths = []
    for filename in os.listdir(img_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            img_paths.append(os.path.join(img_dir, filename))
    return img_paths


def cropIm2size(fimage, fname):
    
    image_size = Resolution()
    image_size.width = 1920
    image_size.height = 1080
    
    # Set the video resolution to HD720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
    serial_number = 12970 # int(sys.argv[1])
    
    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)
    
    red_lower=np.array([0,0,75],np.uint8)
    red_upper=np.array([250,255,255],np.uint8)
    height, width = fimage.shape[:2]
    print (height, width)
    crop_img = fimage[0:1080,500:1580]    #fimage[0:1080,500:1570]
    cv2.imwrite('D:/test_images/' +"tstIm_0" +str(fname) + ".jpg", crop_img)
    retval, frame = cap.read()
    frm=frm+1
    start_t = time.time()
    # Extract left and right images from side-by-side
    left_right_image = np.split(frame, 2, axis=1)
    #left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)
        
    img=right_rect        
    img=img[0:1080,500:1570]            #375:1815     Y=787  X=787  

    
f_dir = 'C:/Users/anasa/Pictures/Camera Roll' #'C:/Users/anasa/Desktop/Datasets/papers'
fpath = ret_path(f_dir)

fname = 1
for image_path in fpath:
    fimage=cv2.imread(image_path)
    cropIm2size(fimage, fname)
    fname+=1
    # for i in range(4):
    #     translateImg(image, fname)
    #     fname+=1
    

# paper_dir = 'C:/Users/anasa/Desktop/Datasets/ForTrain/train/2_cartonCup'
# paper_path = ret_path(paper_dir)

# bot_dir = 'C:/Users/anasa/Desktop/Datasets/ForTrain/train/3_bottles'
# bot_path = ret_path(bot_dir)

# nyl_dir = 'C:/Users/anasa/Desktop/Datasets/ForTrain/train/4_nylon'
# nyl_path = ret_path(nyl_dir)