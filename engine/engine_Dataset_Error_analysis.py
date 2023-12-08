import os
import cv2
import numpy as np

from numpy.lib.arraysetops import unique

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, intersectionAndUnion, find_free_port, check_makedirs, colorize

import torch

# --------------------------------------------------------------------------------
# 统计各类别的像素点总数
def count_class(input, K=256):
    input = input.view(-1)   # 将 tensor 张成一行
    output = torch.histc(input, bins=K, min=0, max=K-1)    # 交集
    return output

# --------------------------------------------------
# 进行单张 img 的统计信息
# 输入为 img 的 Gray GT 图像。 HxWx3
# -----------------
# 进行 GT 图像的信息统计分析
def GT_img_analysis(image_cv2, cfg, logger):

    label = torch.from_numpy(image_cv2)              # np 转 tensor
    label = label.long()                             # label 要转 long 型
    label = label.cuda(non_blocking=True)

    logger.info('图像中出现的类别: {}.'.format(torch.unique(label)))
    h, w = label.size()
    pix = h*w

    output = count_class(label, cfg['BINS'])    # 获得各类别的统计梯度直方图。仅统计 0~cfg['BINS']-1 内的灰度值

    # 各类信息输出--------------
    logger.info('图像大小：{}x{}.'.format(h ,w))
    sum_pix_train = output.sum()                          # 参与训练的像素点个数
    logger.info('总像素点个数: {} .参与训练的总像素点个数: {} .百分比： {}.'.format(pix, sum_pix_train, sum_pix_train/pix))
    logger.info('各类别的像素点分布统计: {}{}.'.format('\n', output))

    # # 不去除 ignore_label 像素
    # percentage_class = output/pix
    # logger.info('各类别的像素点占总像素点的百分比:{}{}'.format('\n', percentage_class))
    # logger.info('总百分比: {}'.format(percentage_class.sum()))

    # 去除掉 ignore label 像素
    percentage_class_train = output/sum_pix_train
    logger.info('各类别的像素点占参与训练的像素点的百分比: {}{}.'.format('\n', percentage_class_train))

# 进行 pred img 的 analysis 时候，需要去除掉 ignore_label 区域
def pred_img_analysis(pred, target, cfg, logger):

    pred = torch.from_numpy(pred)                        # np 转 tensor
    pred = pred.long().cuda(non_blocking=True)           # label 要转 long 型
    
    target = torch.from_numpy(target)                        # np 转 tensor
    target = target.long().cuda(non_blocking=True)           # label 要转 long 型

    pred[target == cfg['ignore_label']] = cfg['ignore_label']    # 去除掉 ignore_label 区域

    logger.info('图像中出现的类别: {}.'.format(torch.unique(pred)))
    h, w = pred.size()
    pix = h*w

    output = count_class(pred, cfg['BINS'])    # 获得各类别的统计梯度直方图。仅统计 0~cfg['BINS']-1 内的灰度值

    # 各类信息输出--------------
    # logger.info('图像大小：{}x{}.'.format(h ,w))
    sum_pix_train = output.sum()                          # 参与训练的像素点个数
    # logger.info('总像素点个数: {} .参与训练的总像素点个数: {} .百分比： {}.'.format(pix, sum_pix_train, sum_pix_train/pix))
    logger.info('各类别的像素点分布统计: {}{}.'.format('\n', output))

    # # 不去除 ignore_label 像素
    # percentage_class = output/pix
    # logger.info('各类别的像素点占总像素点的百分比:{}{}'.format('\n', percentage_class))
    # logger.info('总百分比: {}'.format(percentage_class.sum()))

    # 去除掉 ignore label 像素
    percentage_class_train = output/sum_pix_train
    logger.info('各类别的像素点占参与训练的像素点的百分比: {}{}.'.format('\n', percentage_class_train))

# -----------------------------------------------------------------------------------
# 进行 error 预测的可视化操作
def img_error_analysis(error_save_path, pred, target, cfg):

    pred = torch.from_numpy(pred)                        # np 转 tensor
    pred = pred.long().cuda(non_blocking=True)           # label 要转 long 型
    
    target = torch.from_numpy(target)                        # np 转 tensor
    target = target.long().cuda(non_blocking=True)           # label 要转 long 型

    pred[target == cfg['ignore_label']] = cfg['ignore_label']    # target 中被忽视的像素值在 output 中相应位置

    error_map = pred - target
    error_map[error_map != 0] = 255     # 分类错误的像素点
    error_map[target == cfg['ignore_label']] = 128     # ignore_label 的像素点

    gray_error_map = np.uint8(error_map.cpu()) 
    cv2.imwrite(error_save_path, gray_error_map)

# -----------------------------------------------------------------------------------
# 根据保存的图像结果（label），进行 error 分析
def main_img_error_analysis(data_list, pred_folder, error_folder, classes, names, logger, cfg):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.set_printoptions(
    # precision=4,    # 保留小数点后几位，默认4
    sci_mode=False    # 关闭科学计数，去掉后面 e-3 以使打印 tensor 可以对齐
    )

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        logger.info('--------------------------------------------------------------------------------')
        logger.info('开始进行图片 {} 的统计特性分析及 Error analysis：'.format(image_name+'.png'))
        logger.info('--------------------------------------------------------------------------------')
        
        logger.info('-------------')
        logger.info('开始进行 GT 统计特性分析：')
        logger.info('GT 图片所在路径： {} '.format(target_path))
        GT_img_analysis(target, cfg, logger)

        logger.info('-------------')
        logger.info('开始进行 prey 统计特性分析：')
        pred_img_analysis(pred ,target , cfg, logger)

        intersection, union, target_out = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target_out)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # 单张图片的 average 精度
        img_iou = intersection / union  # 单张图片的 IoU

        logger.info('单张图片 accuracy {:.4f}.'.format(accuracy))
        logger.info('单张图片 iou: {}{}.'.format('\n', torch.from_numpy(img_iou).float()))

        logger.info('-------------')
        error_save_path = os.path.join(error_folder, image_name+'.png')
        logger.info('开始进行 error map 的计算及保存：')
        logger.info('error 图片保存路径： {} '.format(error_save_path))
        img_error_analysis(error_save_path, pred, target, cfg)


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    logger.info('--------------------------------------------------------------------------------')
    logger.info('数据集及单独类别的精度：')
    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))