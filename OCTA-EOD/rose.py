import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split

"""
Dataset: https://imed.nimte.ac.cn/dataofrose.html
https://github.com/iMED-Lab/OCTA-Net-OCTA-Vessel-Segmentation-Network
A. ROSE1: 117 OCTA images from 39 subjects (26 with disease  and the rest are healthy control) and was split into 90 images for training and 27 images for testing. 
          Each subject has en face angiograms of superficial (SVC), deep (DVC), and the inner retinal vascular plexus that includes both SVC and DVC (SVC+DVC) respectively. 
          All the OCTA scans were captured by the RTVue XR Avanti SD-OCT system (Optovue, USA) equipped with AngioVue software, with image resolution of 304 × 304 pixels. 
          The scan area was 3×3 mm2 area centred at the fovea. 
          Two different types of vessel annotations are available: centerline-level and pixel-level annotation. 
B. ROSE2: 112 OCT-A images of 112 eyes acquired by Heidelberg OCT2 system with Spectralis software (Heidelberg Engineering, Heidelberg, Germany) 
          and was split into 90 images for training and 22 images for testing. 
          All the images in this dataset are the en face angiograms of SVC within a 3×3 mm2 area centred at the fovea, 
          and each image was resized into a grayscale image with 840 × 840 pixels.
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, data_type):
        """
        Args: 
        path_to_img_dir: path to image and mask directory.
        data_type: ['ROSE-1/DVC/', 'ROSE-1/SVC/', 'ROSE-1/SVC_DVC/','ROSE-2/']
        """
        imageIDs, maskIDs = [], []
        gt_path = path_to_img_dir + "gt/"
        if data_type == 'ROSE-2/':
            img_path = path_to_img_dir + "original/"
        else:  #"ROSE-1"
            img_path = path_to_img_dir + "img/"

        for root, dirs, files in os.walk(img_path):
            for file in files:
                ID = os.path.join(img_path + file)
                imageIDs.append(ID)
        for root, dirs, files in os.walk(gt_path):
            for file in files:
                ID = os.path.join(gt_path + file)
                maskIDs.append(ID)

        self.imageIDs = imageIDs
        self.maskIDs = maskIDs
        self.transform_octa = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its masks
        """
        image = self.imageIDs[index]
        mask = self.maskIDs[index]

        image = Image.open(image)#.convert('L')
        image = self.transform_octa(image)
        mask = Image.open(mask).convert('L')
        mask = self.transform_octa(mask)
  
        return image, mask

    def __len__(self):
        return len(self.imageIDs)

PATH_TO_IMAGES_DIR_ROOT= '/data/fjsdata/OCTA-Rose/'
def get_train_dataloader(batch_size, shuffle, num_workers, type='ROSE-1/DVC/'):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_ROOT + type + "train/", data_type = type)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_test_dataloader(batch_size, shuffle, num_workers, type='ROSE-1/DVC/'):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_ROOT + type + "test/", data_type = type)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test


if __name__ == "__main__":
    #for debug   
    dataloader_train = get_test_dataloader(batch_size=2, shuffle=True, num_workers=0, type='ROSE-1/SVC_DVC/')
    for batch_idx, (image, mask) in enumerate(dataloader_train):
        print(image.shape)
        print(mask.shape)
        break
