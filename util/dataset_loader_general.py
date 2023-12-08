import os
import os.path
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from util.util_onehot_boundary import *

# ==================================================
# 获得数据集中 img 及相应 label 的 root

def get_img_label(split='train', data_root=None, list_root=None, max_num=None):
    assert split in ['train', 'val', 'test']

    image_label_list = []
    list_read = open(list_root).readlines()   # 读取 train_aug.txt，即训练图片的文件名 ？？？没有这个文件呀？？？
    length_list = len(list_read)

    if max_num is not None:
        list_read = list_read[0:min(max_num,length_list)]
    
    for line in list_read:
        line = line.strip()   # 去除字符串首尾的空格
        line_split = line.split(' ')   # 按空格进行分割
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])   
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            # train 和 val 的 txt 中有一行两列：img 地址，相应 label 地址
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])   # 获取 img 的绝对地址
            label_name = os.path.join(data_root, line_split[1])   # 获取 label 的绝对地址

        item = (image_name, label_name)   # 元组
        image_label_list.append(item)    # 列表
    
    return image_label_list

# ==================================================
class SegDataset(Dataset):
    def __init__(self, data_root=None, list_root=None, split='train', max_num=None, transform=None, num_classes=None, cfg=None):
        self.split = split
        self.transform = transform
        self.img_label_root = get_img_label(split, data_root, list_root, max_num)
        self.num_classes = num_classes
        self.if_get_binary_boundary_mask = cfg['if_get_binary_boundary_mask']

    def __len__(self):
        return len(self.img_label_root)

    def __getitem__(self,index):

        image_path, label_path = self.img_label_root[index]

        # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，读取出来为 BGR 类：H * W * 3
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)   

        # BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        # 读取灰度图
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if self.transform is not None:
            image, label = self.transform(image, label)   # 进行 img、label 的裁剪、翻转等变换

        if self.if_get_binary_boundary_mask:
        # 利用 BFPNet 中计算 boundary 的方法计算 2 值 boundary map
            binary_edgemap = get_binary_boundary(label, self.num_classes)   # 获得 boundary ground truth
        else:
            # binary_edgemap = None  # 传入 None 好像还不行,会报错
            binary_edgemap = image

        return image, label, binary_edgemap

# ==================================================
# 直接计算数据集中所有图片的 label、binary mask 及其 transform：
def get_img_various_mask(split='train', data_root=None, list_root=None, max_num=None, transform=None, num_classes=None):
    
    img_label_root = get_img_label(split, data_root, list_root, max_num)
    outputs = []

    print(len(img_label_root))
    for i in range(len(img_label_root)):

        print(i)
        
        image_path, label_path = img_label_root[i]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)   

        # BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        # 读取灰度图
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if transform is not None:
            image, label = transform(image, label)   # 进行 img、label 的裁剪、翻转等变换

        if split != 'test':
            # 从 tensor mask 中计算 binary edgemap 的方法。
            # 这几段计算 binary_edgemap 的方式会导致 Dataloader 效率低下，进而引起 GPU 利用率低的问题
            _edgemap = label.numpy()
            _edgemap = mask_to_onehot(_edgemap, num_classes)
            _edgemap = onehot_to_binary_edges(_edgemap, 2, num_classes)
            binary_edgemap = torch.from_numpy(_edgemap).float()
        else:
            binary_edgemap = None  

        out = (image, label, binary_edgemap)
        outputs.append(out)

    return outputs

# ==================================================
# 由于一次性在外面加载的数据集过大，会由于内存占用过大导致：
# torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGKILL

# class SegDataset(Dataset):
#     def __init__(self, data_root=None, list_root=None, split='train', max_num=None, transform=None, num_classes=None):
        
#         self.imgs_masks = get_img_various_mask(split, data_root, list_root, max_num, transform, num_classes)
        

#     def __len__(self):
#         return len(self.imgs_masks)

#     def __getitem__(self,index):

#         image, label, binary_edgemap = self.imgs_masks[index]

#         return image, label, binary_edgemap

# ====================================================================

if __name__ == '__main__':
    img_label_root = get_img_label(data_root='/home/zhangyanhua/Code_python/Dataset/VOC2012_augment', list_root='/home/zhangyanhua/Code_python/Dataset/VOC2012_augment/val.txt', split='val',max_num=100000)
    img_root,label_root = img_label_root[0]
    # print(img_root)
    # print(label_root)
    print(len(img_label_root))