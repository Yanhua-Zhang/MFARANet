import os
from tqdm import tqdm
import time
import cv2
import numpy as np

import torch
from torch.nn import functional as F

# from util.HRNet_base_dataset import inference_engine     # 利用 HRNet 中的 inference 方式进行 MS
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, intersectionAndUnion, find_free_port, check_makedirs, colorize




#   定义测试网络
def test_HRNet(cfg, test_loader, data_list, model, test_engine , classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors, logger):
    # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        # 这里利用 tqdm 自动加入一个进度条方便查看
        for i, data in enumerate(tqdm(test_loader)):
            data_time.update(time.time() - end)      # 已训练数据集的总训练时间

            image = data[0]
            size = image.size()  # 1, 3, H, W
            
            # 选择 MS、SS、或 full_image_inference。
            if not cfg['full_image_inference']:
                # 这里换为根据 PSP 改良版的 HRNet
                prediction = test_engine.multi_scale_inference_PSP(
                    cfg,
                    model,
                    image,
                    scales=cfg['scales'],
                    flip=cfg['flip'])         # SS inference 的 flip 应该为 false
            else:
                prediction = test_engine.inference(model, image, flip=False)

            if prediction.size()[-2] != size[0] or prediction.size()[-1] != size[1]:
                prediction = F.interpolate(
                    prediction, size[-2:],
                    mode='bilinear', align_corners=True
                )

            prediction = prediction.max(1)[1].squeeze(0)  # 去掉第一个维度
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % 100 == 0) or (i + 1 == len(test_loader)):                                         
                logger.info('Test: [{}/{}] '
                            '加载一个 test 图像的总时间 {data_time.val:.3f} (加载一个 test 图像的平均时间 {data_time.avg:.3f}) '
                            '当前图像的 inference 时间 {batch_time.val:.3f} (已经 test 图像的平均 inference 时间 {batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                        data_time=data_time,
                                                                                        batch_time=batch_time))
            # 检查是否存在文件夹，不存在进行生成
            # check_makedirs(gray_folder)
            # check_makedirs(color_folder)

            gray = np.uint8(prediction.cpu())

            # print(gray.shape)

            if cfg['NAME_dataset'] == 'cityscapes':
                
                if gray.shape != (1024, 2048):
                    #  cv2 needs a (w, h) input: (2048, 1024)
                    gray = cv2.resize(gray, (2048, 1024), interpolation=cv2.INTER_NEAREST)   

            # print(gray.shape)

            color = colorize(gray, colors)   # label 到可视化 annotion 的转换。利用 colorize 函数转换为 PIL Image 格式
            image_path, _ = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            gray_path = os.path.join(gray_folder, image_name + '.png')
            color_path = os.path.join(color_folder, image_name + '.png')
            cv2.imwrite(gray_path, gray)
            color.save(color_path)
    # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
