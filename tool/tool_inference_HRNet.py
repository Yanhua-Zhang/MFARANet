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

from util.HRNet_base_dataset import inference_engine     # 利用 HRNet 中的 inference 方式进行 MS
from engine.egnine_test_HRNet import test_HRNet

from engine.engine_test import test_PSP, cal_acc
from engine.engine_Dataset_Error_analysis import main_img_error_analysis

# 加载网络结构，并实例化
from model.model_loader_general import load_model   
# ==================================================================

def main_inference(cfg):
    
    # =====================================================================
    # # 读取 yaml 文件
    # yaml_path = '/home/zhangyanhua/Code_python/Semantic-seg-multiprocessing-general/config/cityscapes/SFNet/cityscapes_sfnet18.yaml'
    # cfg = get_args(yaml_path)
    if cfg['load_trained_model']:
        cfg['load_checkpoint'] = cfg['load_trained_model']
    else:
        path_model = './save/model/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + str(cfg['epochs']) + '.pth'
        cfg['load_checkpoint'] = path_model # 加载保存 model 的路径
    # =====================================================================
    # 建立记录 MS、SS inference 的日志
    
    check_makedirs('./save/log/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])  # 检查保存 log 的文件夹是否存在，不存在进行生成
    file_name_log = './save/log/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset']+'/'+"record_"+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset']  +"_train.log"
    
    logger = build_logger(cfg['logger_name_trainval'], file_name_log)
    logger.info('Inference =======================================================================================')

    # 建立记录 error analysis 的日志
    log_error_analysis_path = './save/log/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset']+'/'+"record_"+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset']  +"_error_analysis.log"
    logger_error = build_logger(cfg['logger_name_error_analysis'], log_error_analysis_path)
    
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
    # 限制使用的GPU个数, 设置:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg['train_gpu'])  
    # =====================================================================
    # 加载数据集
    value_scale = cfg['value_scale']
    mean = cfg['mean']
    mean = [item * value_scale for item in mean]
    std = cfg['std']
    std = [item * value_scale for item in std]

    # test_transform = transform.Compose([transform.ToTensor()])    # 只对测试图像做了一个 ToTensor 的操作

    # 使用 HRNet 进行测试时，需要进行 toTensor 和标准化操作
    test_transform = transform.Compose([
        # np.ndarray 转 tensor
        transform.ToTensor(),
        # 标准化
        transform.Normalize(mean=mean, std=std)])    

    data_test = dataset_loader_general.SegDataset(data_root=cfg['data_root'],
                    list_root=cfg['test_list_root'],split=cfg['split'],max_num=cfg['max_num'],
                    transform=test_transform, num_classes=cfg['num_classes'],
                    cfg=cfg)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size= cfg['batch_size_test'], shuffle=False, # 洗牌
                                                num_workers=cfg['num_workers'], pin_memory=True)  # 子进程，复制数据到CUDA

    # =============================================================
    # 保存可视化结果
    save_path = './save/image/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] 
    check_makedirs(save_path)                        # 检查文件夹是否存在
    gray_folder_base = os.path.join(save_path, 'gray')    # 保存可视化结果的文件夹
    color_folder_base = os.path.join(save_path, 'color')
    error_folder_base = os.path.join(save_path, 'error')
    check_makedirs(gray_folder_base)                    # 检查文件夹是否存在
    check_makedirs(color_folder_base)                   # 检查文件夹是否存在
    check_makedirs(error_folder_base) 

    # 从 txt 文件中读取类别对应的像素值、类别名
    colors_path = './data/'+ cfg['NAME_dataset']+'/'+ cfg['NAME_dataset'] + '_colors.txt'
    names_path = './data/'+ cfg['NAME_dataset']+'/'+ cfg['NAME_dataset'] + '_names.txt'
    
    # -------------------------------------------------------------
    # 进行 MS inference_engine 的实例化
    engine = inference_engine(
                 ignore_label=cfg['ignore_label'],
                 base_size=cfg['base_size'],
                 crop_size=(cfg['test_h'], cfg['test_w']),    # 这里应该是 cfg['test_h'], cfg['test_w']。不过这样也方便。
                 downsample_rate=1,
                 scale_factor=16,
                 mean=cfg['mean'],
                 std=cfg['std']) 


    # =============================================================
    if cfg['if_MS_inference']:
        cfg['full_image_inference'] = False
        cfg['flip'] = True
        cfg['stride_rate'] = 2/3

        logger.info('---------------->开始进行 MS inference')

        gray_folder_MS = os.path.join(gray_folder_base, 'MS')
        color_folder_MS = os.path.join(color_folder_base, 'MS')    
        check_makedirs(gray_folder_MS)                    # 检查文件夹是否存在
        check_makedirs(color_folder_MS)                   # 检查文件夹是否存在
        

        # # 得到 id img,并保存
        if cfg['if_save_image_result']:
            colors = np.loadtxt(colors_path).astype('uint8')
            test_HRNet(cfg, test_loader, data_test.img_label_root, model, engine, 
                    cfg['num_classes'], mean, std, cfg['base_size'], cfg['test_h'], cfg['test_w'], cfg['scales'], 
                    gray_folder_MS, color_folder_MS, colors, logger)

        # 根据 id img，计算 mIoU、acc 指标
        if cfg['if_get_acc_from_image_result']:
            logger.info('------------>get_acc_from_image_result')
            names = [line.rstrip('\n') for line in open(names_path)]
            cal_acc(data_test.img_label_root, gray_folder_MS, cfg['num_classes'], names, logger)


        # 根据保存的图像结果（label），进行 error 分析
        if cfg['if_error_analysis']:
            logger_error.info('------------>开始进行 MS 结果的 error analysis')
            error_folder_MS = os.path.join(error_folder_base, 'MS')
            check_makedirs(error_folder_MS)

            names = [line.rstrip('\n') for line in open(names_path)]
            main_img_error_analysis(data_test.img_label_root, gray_folder_MS, error_folder_MS, cfg['num_classes'], names, logger_error, cfg)

    # # ----------------------------------------------------------------
    # 进行 SS_flip inference 
    if cfg['if_single_inference']: 
        cfg['scales'] = [1.0]
        cfg['full_image_inference'] = False
        cfg['flip'] = True   # SS 时，不进行 Flip
        cfg['stride_rate'] = 1
        logger.info('---------------->开始进行 SS_flip inference')

        gray_folder_SS_flip = os.path.join(gray_folder_base, 'SS_flip')
        color_folder_SS_flip = os.path.join(color_folder_base, 'SS_flip')
        check_makedirs(gray_folder_SS_flip)                    # 检查文件夹是否存在
        check_makedirs(color_folder_SS_flip)                   # 检查文件夹是否存在

        # #得到 id img,并保存
        if cfg['if_save_image_result']:
            colors = np.loadtxt(colors_path).astype('uint8')
            test_HRNet(cfg, test_loader, data_test.img_label_root, model, engine, 
                    cfg['num_classes'], mean, std, cfg['base_size'], cfg['test_h'], cfg['test_w'], cfg['scales'], 
                    gray_folder_SS_flip, color_folder_SS_flip, colors, logger)

        # 根据 id img，计算 mIoU、acc 指标
        if cfg['if_get_acc_from_image_result']:
            logger.info('------------>get_acc_from_image_result')
            names = [line.rstrip('\n') for line in open(names_path)]
            cal_acc(data_test.img_label_root, gray_folder_SS_flip, cfg['num_classes'], names, logger)

        # 根据保存的图像结果（label），进行 error 分析
        if cfg['if_error_analysis']:
            logger_error.info('------------>开始进行 SS 结果的 error analysis')
            error_folder_SS_flip = os.path.join(error_folder_base, 'SS_flip')
            check_makedirs(error_folder_SS_flip)

            names = [line.rstrip('\n') for line in open(names_path)]
            main_img_error_analysis(data_test.img_label_root, gray_folder_SS_flip, error_folder_SS_flip, cfg['num_classes'], names, logger_error, cfg)

    # # ----------------------------------------------------------------
    # 进行 SS 
    if cfg['if_single_inference']: 
        cfg['scales'] = [1.0]
        cfg['full_image_inference'] = False
        cfg['flip'] = False   # SS 时，不进行 Flip
        cfg['stride_rate'] = 1
        logger.info('---------------->开始进行 SS inference')

        gray_folder_SS = os.path.join(gray_folder_base, 'SS')
        color_folder_SS = os.path.join(color_folder_base, 'SS')
        check_makedirs(gray_folder_SS)                    # 检查文件夹是否存在
        check_makedirs(color_folder_SS)                   # 检查文件夹是否存在

        # #得到 id img,并保存
        if cfg['if_save_image_result']:
            colors = np.loadtxt(colors_path).astype('uint8')
            test_HRNet(cfg, test_loader, data_test.img_label_root, model, engine, 
                    cfg['num_classes'], mean, std, cfg['base_size'], cfg['test_h'], cfg['test_w'], cfg['scales'], 
                    gray_folder_SS, color_folder_SS, colors, logger)

        # 根据 id img，计算 mIoU、acc 指标
        if cfg['if_get_acc_from_image_result']:
            logger.info('------------>get_acc_from_image_result')
            names = [line.rstrip('\n') for line in open(names_path)]
            cal_acc(data_test.img_label_root, gray_folder_SS, cfg['num_classes'], names, logger)

        # 根据保存的图像结果（label），进行 error 分析
        if cfg['if_error_analysis']:
            logger_error.info('------------>开始进行 SS 结果的 error analysis')
            error_folder_SS = os.path.join(error_folder_base, 'SS')
            check_makedirs(error_folder_SS)

            names = [line.rstrip('\n') for line in open(names_path)]
            main_img_error_analysis(data_test.img_label_root, gray_folder_SS, error_folder_SS, cfg['num_classes'], names, logger_error, cfg)

    # # ----------------------------------------------------------------
    # 进行 full_image_inference：SFNet、DDRNet 中使用的 full image inferences。
    cfg['full_image_inference'] = True
    if cfg['full_image_inference']: 
        # cfg['scales'] = [1.0]
        cfg['flip'] = False   # SS 时，不进行 Flip
        logger.info('---------------->开始进行 full image inferences')

        gray_folder_full_img = os.path.join(gray_folder_base, 'full_img')
        color_folder_full_img = os.path.join(color_folder_base, 'full_img')
        check_makedirs(gray_folder_full_img)                    # 检查文件夹是否存在
        check_makedirs(color_folder_full_img)                   # 检查文件夹是否存在

        # #得到 id img,并保存
        if cfg['if_save_image_result']:
            colors = np.loadtxt(colors_path).astype('uint8')
            test_HRNet(cfg, test_loader, data_test.img_label_root, model, engine, 
                    cfg['num_classes'], mean, std, cfg['base_size'], cfg['test_h'], cfg['test_w'], cfg['scales'], 
                    gray_folder_full_img, color_folder_full_img, colors, logger)

        # 根据 id img，计算 mIoU、acc 指标
        if cfg['if_get_acc_from_image_result']:
            logger.info('------------>get_acc_from_image_result')
            names = [line.rstrip('\n') for line in open(names_path)]
            cal_acc(data_test.img_label_root, gray_folder_full_img, cfg['num_classes'], names, logger)

        # 根据保存的图像结果（label），进行 error 分析
        if cfg['if_error_analysis']:
            logger_error.info('------------>开始进行 full_img 结果的 error analysis')
            error_folder_full_img = os.path.join(error_folder_base, 'full_img')
            check_makedirs(error_folder_full_img)

            names = [line.rstrip('\n') for line in open(names_path)]
            main_img_error_analysis(data_test.img_label_root, gray_folder_full_img, error_folder_full_img, cfg['num_classes'], names, logger_error, cfg)

# ===================================================================================
if __name__ == '__main__':
    main_inference()