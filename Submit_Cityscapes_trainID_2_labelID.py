import os
import numpy as np
from PIL import Image
import cv2
from numpy.lib.arraysetops import unique
from tqdm import tqdm
import torch

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs
# 利用 label map 将 trainID 映射到 lavelID：

# ------------------------------------------------------
ignore_label = -1
label_dict = {-1: ignore_label, 0: ignore_label,
              1: ignore_label, 2: ignore_label,
              3: ignore_label, 4: ignore_label,
              5: ignore_label, 6: ignore_label,
              7: 0, 8: 1, 9: ignore_label,
              10: ignore_label, 11: 2, 12: 3,
              13: 4, 14: ignore_label, 15: ignore_label,
              16: ignore_label, 17: 5, 18: ignore_label,
              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
              25: 12, 26: 13, 27: 14, 28: 15,
              29: ignore_label, 30: ignore_label,
              31: 16, 32: 17, 33: 18}

# =======================================================
key_list = list(label_dict.keys())      # 字典转列表
value_list = list(label_dict.values())

key_tensor = torch.tensor(key_list).int()    # 列表转 tensor
value_tensor = torch.tensor(value_list).int()

# out = key_list[value_list.index(0)]


def trainID2labelID_V2(img):
    img_torch = torch.tensor(img).int()  # np 转 tensor
    
    out = torch.ones(img_torch.size())*255
    out = out.int()

    # 注意这个 reversed 很重要，不然前面改过的值会被反复改
    for i in range(19):
        out[img_torch == i] = key_tensor[value_tensor == i]

    out = out.cpu().numpy()

    return out
# ------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_Name', type=str, default='MFARANetScaleChoice_resnet_18_deep_stem_Branch_1_2_3_4_R1_Drop_0.05_best_val_test_cityscapes', help='path')

args = parser.parse_args()

# ------------------------------------------------------------------------
if __name__ == '__main__':

    #--------------------------------------------------------------------
    path_MS = './save/image/' + args.path_Name + '/gray/MS'

    save_path_MS = path_MS + '_labelID'

    # 检查保存 save_path 的文件夹是否存在，不存在进行生成
    check_makedirs(save_path_MS)     

    #-------------------------------------------
    # 进行批量操作
    list_read = os.listdir(path_MS)  # 列出文件夹下所有的目录与文件
    print('统计的总图片张数: {}'.format(len(list_read)))

    for i in tqdm(range(len(list_read))):
    # for i in range(2):
        # 获取 img 文件的绝对地址
        image_path = os.path.join(path_MS, list_read[i])
        image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        id_image = trainID2labelID_V2(image_cv2)  # train_id 2 label_id

        save_name = os.path.join(save_path_MS, list_read[i])
        cv2.imwrite(save_name, id_image)


    #--------------------------------------------------------------------
    path_full_img= './save/image/' + args.path_Name + '/gray/full_img'

    save_path_full_img = path_full_img + '_labelID'

    # 检查保存 save_path 的文件夹是否存在，不存在进行生成
    check_makedirs(save_path_full_img)     

    #-------------------------------------------
    # 进行批量操作
    list_read = os.listdir(path_full_img)  # 列出文件夹下所有的目录与文件
    print('统计的总图片张数: {}'.format(len(list_read)))

    for i in tqdm(range(len(list_read))):
    # for i in range(2):
        # 获取 img 文件的绝对地址
        image_path = os.path.join(path_full_img, list_read[i])
        image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        id_image = trainID2labelID_V2(image_cv2)  # train_id 2 label_id

        save_name = os.path.join(save_path_full_img, list_read[i])
        cv2.imwrite(save_name, id_image)