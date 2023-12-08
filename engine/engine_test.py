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


from util import  transform                         # 加载自己写的 dataset_loader、transform 模块
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, intersectionAndUnion, find_free_port, check_makedirs, colorize

def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()  # HxWx3  --> 3xHxW
    # 归一化
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()

    # flip：沿给定轴颠倒数组中元素的顺序
    if flip:
        input = torch.cat([input, input.flip(3)], 0)    # 沿某一维度进行拼接

    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    
    if flip:
        output = (output[0] + output[1].flip(2)) / 2         # 没看懂
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape  # 尺度变换后的图像大小

    # 如果图像小于 crop_h, crop_w，进行 0 填充
    pad_h = max(crop_h - ori_h, 0)   # 需要进行 0 填充的个数
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape
    # 滑窗的步长
    stride_h = int(np.ceil(crop_h*stride_rate))  # ceil 向上取整、floor 向下取整
    stride_w = int(np.ceil(crop_w*stride_rate))
    # 窗长、窗宽
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)

    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()  # 取图像中的一个窗（滑块儿）
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)   # np.expand_dims：扩充维度
    
    # 注意！！！前面进行 0 填充后，这里要去掉 0 填充
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]   # 去掉 0 padding


    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)   # 线性插值到原图像大小
    return prediction

#   定义测试网络
def test_PSP(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors, logger):
    # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)

        input = np.squeeze(input.numpy(), axis=0)  # 去掉一个维度
        image = np.transpose(input, (1, 2, 0))     # HxWx3
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)

        #  Multi-scale testing scheme
        # 各个尺度下进行预测，然后融合
        for scale in scales:
            long_size = round(scale * base_size)  # 最近的整数。尺度变换后的图像大小
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)   # 双线性插值
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)   # 将各尺度下的预测结果平均
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 100 == 0) or (i + 1 == len(test_loader)):                                         
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        # 检查是否存在文件夹，不存在进行生成
        # check_makedirs(gray_folder)
        # check_makedirs(color_folder)

        gray = np.uint8(prediction)
        color = colorize(gray, colors)   # label 到可视化 annotion 的转换。利用 colorize 函数转换为 PIL Image 格式
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


# 根据保存的图像结果（label）计算 mIoU 等指标
def cal_acc(data_list, pred_folder, classes, names, logger):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        if ((i + 1) % 200 == 0): 
            logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

#   定义测试网络
def direct_cal_acc_PSP(test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales, logger):
    # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()

    intersection_sum = 0   
    union_sum = 0
    target_sum = 0
    
    intersection, union, target = 0, 0, 0

    model.eval()
    end = time.time()
    for i, data in enumerate(test_loader):
        data_time.update(time.time() - end)

        input, label = data[0], data[1]
        input = np.squeeze(input.numpy(), axis=0)  # 去掉一个维度
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)

        #  Multi-scale testing scheme
        # 各个尺度下进行预测，然后融合
        for scale in scales:
            long_size = round(scale * base_size)  # 最近的整数。尺度变换后的图像大小
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)   # 双线性插值
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)   # 将各尺度下的预测结果平均
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 500 == 0) or (i + 1 == len(test_loader)):                                         
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        
        label = np.squeeze(label.numpy(), axis=0)  # 去掉一个维度
        # label = np.transpose(label, (1, 2, 0))   # label 变为 h,w,1
        intersection, union, target = intersectionAndUnion(prediction, label, classes)

        # 一个 epoch 下的统计值
        intersection_sum += intersection   
        union_sum += union
        target_sum += target

        if ((i + 1) % 100 == 0): 
            acc_class_epoch = intersection_sum / (target_sum + 1e-10)   # 类别精度
            acc_all_epoch = sum(intersection_sum) / (sum(target_sum) + 1e-10)
            acc_mean_epoch = np.mean(acc_class_epoch)                         # 类别平均精度

            IoU_class_epoch = intersection_sum / (union_sum + 1e-10)   # 类别精度
            IoU_mean_epoch = np.mean(IoU_class_epoch)   

            logger.info('Test: [{}/{}] Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(i + 1, len(test_loader), IoU_mean_epoch, acc_mean_epoch, acc_all_epoch))
            # for j in range(classes):
            #     logger.info('Test: [{}/{}] Class_{} result: iou/accuracy {:.4f}/{:.4f}.'.format(i + 1, len(test_loader), j, IoU_class_epoch[j], acc_class_epoch[j]))


    acc_class = intersection_sum / (target_sum + 1e-10)   # 类别精度
    acc_all = sum(intersection_sum) / (sum(target_sum) + 1e-10)
    acc_mean = np.mean(acc_class)                         # 类别平均精度

    IoU_class = intersection_sum / (union_sum + 1e-10)   # 类别精度
    IoU_mean = np.mean(IoU_class)   

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(IoU_mean, acc_mean, acc_all))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}.'.format(i, IoU_class[i], acc_class[i]))
  

#   定义测试网络
def test_demo(image_path, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors, logger):
    # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order

    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)

    end = time.time()
    #  Multi-scale testing scheme
    # 各个尺度下进行预测，然后融合
    for scale in scales:
        long_size = round(scale * base_size)  # 最近的整数。尺度变换后的图像大小
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)
        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)   # 双线性插值
        prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
    prediction /= len(scales)   # 将各尺度下的预测结果平均
    prediction = np.argmax(prediction, axis=2)
    time_spend = time.time() - end
    logger.info('Inference time: {:.4f}.'.format(time_spend))
  
    
    gray = np.uint8(prediction)
    color = colorize(gray, colors)   # label 到可视化 annotion 的转换。利用 colorize 函数转换为 PIL Image 格式

    image_name = image_path.split('/')[-1].split('.')[0]
    gray_path = os.path.join(gray_folder, image_name + '.png')
    color_path = os.path.join(color_folder, image_name + '.png')
    cv2.imwrite(gray_path, gray)
    color.save(color_path)
    logger.info("=> Prediction saved in {}".format(color_path))
# logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')