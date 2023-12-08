import os
import os.path
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch


# --------------------------------------------------
# 通过对 semantic mask 进行滤波操作，获得 GT boundary mask
# 利用 conv 操作获得 boundary ground truth
def get_boundary(mask, num_classes):
    mask = mask.cpu()
    maskb = np.array(mask).astype('int32')
    # maskb [maskb == 255] = -1
    maskb_ = np.array(mask).astype('float32')
    kernel = np.ones((9,9),np.float32)/81       # trimap: 18
    mask_tmp = cv2.filter2D(maskb_,-1, kernel)
    mask_tmp = abs(mask_tmp - maskb_)
    mask_tmp = mask_tmp > 0.005
    maskb[mask_tmp] = num_classes        # boundry 作为第 num_classes+1 类

    maskb = torch.from_numpy(maskb)
    maskb = maskb.long()   # label 要转 long 型！！！这里要注意
    return maskb

def get_batch_boundary(mask, num_classes):
    batch = mask.size(0)
    batch_boundary = torch.ones_like(mask)
    for i in range(batch):
        batch_boundary[i] = get_boundary(mask[i], num_classes)
    
    return batch_boundary.type_as(mask)