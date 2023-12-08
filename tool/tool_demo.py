import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch import optim, nn

from util import dataset_loader_general, transform, read_yaml          # 加载自己写的 dataset_loader、transform 模块
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

from engine.engine_test import test_demo

# 加载网络结构，并实例化
from model.model_loader_general import load_model   
# ==================================================================

def main_demo(cfg):
    
    # =====================================================================
    # # 读取 yaml 文件
    # yaml_path = '/home/zhangyanhua/Code_python/Semantic-seg-multiprocessing-general/config/cityscapes/SFNet/cityscapes_sfnet18.yaml'
    # cfg = get_args(yaml_path)

    path_model = cfg['absolute_path'] +cfg['FILE_NAME']+'/save/model/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + str(cfg['epochs']) + '.pth'
    cfg['load_checkpoint'] = path_model # 加载保存 model 的路径
    # =====================================================================
    # 建立日志
    
    check_makedirs(cfg['absolute_path']+cfg['FILE_NAME']+"/save/log/"+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])  # 检查保存 log 的文件夹是否存在，不存在进行生成
    file_name_log = cfg['absolute_path']+cfg['FILE_NAME']+"/save/log/"+cfg['NAME_model'] +'_'+ cfg['NAME_dataset']+'/'+"record_"+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset']  +"_train.log"

    logger = build_logger(file_name_log)
    logger.info('Demo test =======================================================================================')
    # =====================================================================
    # 限制使用的GPU个数, 如使用第0和第1编号的GPU,设置:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg['train_gpu'])  
    # =====================================================================
    # 加载模型
    criterion = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label']) 
    # 加载网络结构，并实例化      
    model, _, _ = load_model(cfg, criterion, 'test')

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True  # 可以提升效率

    if os.path.isfile(cfg['load_checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(cfg['load_checkpoint']))
            checkpoint = torch.load(cfg['load_checkpoint'])   # 加载网络模型
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(cfg['load_checkpoint']))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg['load_checkpoint']))
    # =============================================================
    # 加载数据集
    value_scale = cfg['value_scale']
    mean = cfg['mean']
    mean = [item * value_scale for item in mean]
    std = cfg['std']
    std = [item * value_scale for item in std]

    test_transform = transform.Compose([transform.ToTensor()])    # 只对测试图像做了一个 ToTensor 的操作

    data_test = dataset_loader_general.SegDataset(data_root=cfg['data_root'],
                    list_root=cfg['test_list_root'],split='val',max_num=cfg['max_num'],transform=test_transform)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size= cfg['batch_size_test'], shuffle=False, # 洗牌
                                                num_workers=cfg['num_workers'], pin_memory=True)  # 子进程，复制数据到CUDA

    # =============================================================
    # 保存可视化结果
    save_path = cfg['absolute_path']+ cfg['FILE_NAME'] +'/save/demo/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] 
    check_makedirs(save_path)                        # 检查文件夹是否存在
    gray_folder = os.path.join(save_path, 'gray')    # 保存可视化结果的文件夹
    color_folder = os.path.join(save_path, 'color')
    check_makedirs(gray_folder)                    # 检查文件夹是否存在
    check_makedirs(color_folder)                   # 检查文件夹是否存在

    # 从 txt 文件中读取类别对应的像素值、类别名
    colors_path = cfg['absolute_path'] + cfg['FILE_NAME'] +'/data/'+ cfg['NAME_dataset']+'/'+ cfg['NAME_dataset'] + '_colors.txt'
    # names_path = cfg['absolute_path'] + cfg['FILE_NAME'] +'/data/'+ cfg['NAME_dataset']+'/'+ cfg['NAME_dataset'] + '_names.txt'
    
    # =============================================================

    # 得到 id img,并保存
    
    logger.info('------------>save_image_result')
    colors = np.loadtxt(colors_path).astype('uint8')
    test_demo(cfg['demo_image_path'], model, cfg['num_classes'], 
            mean, std, cfg['demo_base_size'], cfg['demo__h'], cfg['demo__w'], cfg['demo_scales'], gray_folder, color_folder, colors, logger)

# ===================================================================================
if __name__ == '__main__':
    main_demo()