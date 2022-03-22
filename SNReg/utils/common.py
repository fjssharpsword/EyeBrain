# encoding: utf-8
"""
Training implementation for CIFAR10 dataset  
Author: Jason.Fang
Update time: 08/07/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def count_bytes(file_size):
    '''
    Count the number of parameters in model
    '''
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def strofsize(integer, remainder, level):
        if integer >= 1024:
            remainder = integer % 1024
            integer //= 1024
            level += 1
            return strofsize(integer, remainder, level)
        else:
            return integer, remainder, level

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    integer, remainder, level = strofsize(int(file_size), 0, 0)
    if level+1 > len(units):
        level = -1
    return ( '{}.{:>03d} {}'.format(integer, remainder, units[level]) )

def compute_AUCs(gt, pred, N_CLASSES):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def dice_coeff(input, target):

    N = target.size(0)
    smooth = 1
    
    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)
    
    intersection = input_flat * target_flat
    
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    dice = loss.sum() / N

    return dice


def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        imageio.imwrite(str(index)+".png", feature_map[index-1])
    plt.savefig('/data/pycode/LungCT3D/imgs/fea_map1.jpg')

def transparent_back(img, gt=True):
    #img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0)) #alpha channel: 0~255
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
            else: 
                if gt: #true mask
                    color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                    img.putpixel(dot,color_1)
                else: #pred mask
                    color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                    img.putpixel(dot,color_1)
    return img

if __name__ == "__main__":
    #for debug  
    A = torch.rand((8, 2048, 64*64))
    B = torch.rand((8, 2048, 64*64)).permute(0, 2, 1)
    start = time.time()
    C = torch.bmm(A, B)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)
    #print('Matrix multiplication completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    A = torch.rand((8, 2048,1))
    B = torch.rand((8, 2048,1)).permute(0, 2, 1)
    start = time.time()
    C = torch.bmm(A, B)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)

    A = torch.rand((8, 2048, 2048))
    B = torch.rand((8, 2048, 1024)) 
    start = time.time()
    C = torch.bmm(A, B)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)

    A_u = torch.rand((8, 2048, 1))
    A_v = torch.rand((8, 2048, 1)).permute(0, 2, 1)
    B = torch.rand((8, 2048, 1024)) 
    start = time.time()
    C = torch.bmm(A_v, B)
    C = torch.bmm(A_u, C)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)