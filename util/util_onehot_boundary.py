import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

import cv2
import torch
# ===================================================================================
# 这里与 GSCNN 中的计算方式一致
# 这里 cityscapes 的第 0 类类似一个背景类，因此其 boundary 不进行计算
# 第 0 类的 boundary 会包含 void label 的边界，因此可能会对训练产生影响
def mask_to_onehot(mask, num_classes):
    _mask = [mask == (i + 1) for i in range(num_classes)]  
    return np.array(_mask).astype(np.uint8)

def onehot_to_binary_edges(mask, radius, num_classes):        
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

# ===================================================================================
# 利用滤波的方式得到 binary boundary
# 根据 BFPNet 改写

def get_binary_boundary(mask, num_classes):
    maskb = np.array(mask).astype('int32')
    # maskb [maskb == 255] = -1
    maskb_ = np.array(mask).astype('float32')
    kernel = np.ones((5,5),np.float32)/25       # trimap: 18
    mask_tmp = cv2.filter2D(maskb_,-1, kernel)
    mask_tmp = abs(mask_tmp - maskb_)
    mask_tmp = mask_tmp > 0.005
    maskb[mask_tmp] = num_classes        # boundry 作为第 num_classes+1 类

    maskb = torch.from_numpy(maskb)

    # 获得 2 值 boundary map
    ones = torch.ones_like(maskb)
    zeros  = torch.zeros_like(maskb)
    binary_mask = torch.where(maskb == num_classes, ones, zeros)

    binary_mask = binary_mask.float()  # label 要转 long 型！！！这里要注意。但计算 2 值 loss，GSCNN 中用的 float。这里使用 float 型。
    binary_mask = torch.unsqueeze(binary_mask, 0)    # 需要增加一个维度
    return binary_mask
