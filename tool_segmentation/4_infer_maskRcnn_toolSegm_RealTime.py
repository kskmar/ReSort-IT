# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:45:48 2020

@author: maria
"""

import os
import sys
import cv2

ROOT_DIR = '/home/microralp/mk_dev/ReSort-IT/mask_rcnn'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist.'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.model as modellib

maskRcnnfolder = '/home/microralp/mk_dev/ReSort-IT/mask_rcnn/'
latesth5 = "toolSegm_X02.h5" 


class InstrumentSegmConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "InstrumentSegm"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 50

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 50
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = "resnet50" #'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) #(16, 32, 64, 128, 256)   #(4, 8, 16, 32, 64)  (32, 64, 128, 256, 320!!!)
    TRAIN_ROIS_PER_IMAGE = 128 #256 #128  #32
    MAX_GT_INSTANCES = 3
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    DETECTION_MIN_CONFIDENCE = 0.9

# ####################################################
# #     ====================================         #
# #         Prepare to run Inference                 #
# #     ====================================         #  
# ####################################################

class InferenceConfig(InstrumentSegmConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #IMAGE_MIN_DIM = 256           s
    #IMAGE_MAX_DIM = 256
    DETECTION_MIN_CONFIDENCE = 0.62

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, latesth5)
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


####################################################
#     ====================================         #
#                Run Inference                     #
#     ====================================         #  
####################################################

#for real time
if __name__ == '__main__':
      class_names = ['BG', 'Tool']

      config = InferenceConfig()
      config.display()

      model.load_weights(model_path, by_name=True)
      colors = visualize.random_colors(len(class_names))

      cap = cv2.VideoCapture(0)
      while True:

          _, frame = cap.read()
          predictions = model.detect([frame],
                                    verbose=1)  # We are replicating the same image to fill up the batch_size
          p = predictions[0]

          output = visualize.display_instances(frame, p['rois'], p['masks'], p['class_ids'],
                                      class_names, p['scores'], colors=colors, real_time=True)
          cv2.imshow("Mask RCNN", output)
          k = cv2.waitKey(10)
          if k & 0xFF == ord('q'):
              break
      cap.release()
      cv2.destroyAllWindows()
