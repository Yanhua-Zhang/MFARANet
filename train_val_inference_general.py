import numpy as np
# import sys
import cv2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import time
import random
# import datetime
# import argparse
# import logging
from tensorboardX import SummaryWriter
import warnings

import torch
from torch import optim, nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# import torchvision
# from torchvision import transforms
# from torch.autograd import Variable
import torch.nn.parallel
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data


# ====================================================================
warnings.filterwarnings("ignore")  # 忽略警告

# 关闭Opencv的多线程。否则 pytorch 的多线程会与 OpenCV 的锁死
cv2.ocl.setUseOpenCL(False)  
cv2.setNumThreads(0)

# ====================================================================
# 加载 engine:
from engine.engine_val import validater
from engine.engine_train import trainer

# 加载 data_loader 及 评估函数
from util import dataset_loader_general, transform, read_yaml          # 加载自己写的 dataset_loader、transform 模块
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

from model.model_loader_general import load_model     

from loss.loss_loader import get_criterion

# from tool.tool_inference import main_inference
from tool.tool_inference_HRNet import main_inference
from tool.tool_demo import main_demo

# =====================================================================


# 如果多线程的话，必须在 rank==0 的进程内保存参数
def main_process():
    return not cfg['multiprocessing_distributed'] or (cfg['multiprocessing_distributed'] and cfg['rank'] % cfg['ngpus_per_node'] == 0)

def main(cfg):

    # =====================================================================
    check_makedirs('./save/log/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])            # 检查保存 log 的文件夹是否存在，不存在进行生成
    check_makedirs('./save/tensorboardX/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])   # 检查文件夹是否存在，不存在生成
    check_makedirs('./save/tensorboardX/compare/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])   # 检查文件夹是否存在，不存在生成
    check_makedirs('./save/model/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])        # 检查保存模型的文件夹是否存在，不存在生成
    check_makedirs('./save/model_best/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])   # 检查保存模型的文件夹是否存在，不存在生成
    # ====================================================================
    if cfg['seed_value'] is not None:
        random.seed(cfg['seed_value'])                 # random 模块中的随机种子，random是python中用于产生随机数的模块
        np.random.seed(cfg['seed_value'])              # numpy 中的随机种子
        torch.manual_seed(cfg['seed_value'])           # pytorch 中的随机种子
        torch.cuda.manual_seed(cfg['seed_value'])      # 设置GPU生成随机数的种子
        torch.cuda.manual_seed_all(cfg['seed_value'])  # 为所有的GPU设置种子
        cudnn.benchmark = False                        # 自动设置最合适的卷积算法，来进行加速（PSP的官方代码在 test 中打开，在 val 中关闭）  
        cudnn.deterministic = True                     # 设为 True 的话，每次返回的卷积算法将是确定的，即默认算法。

    # -------------------------------------------------
    # 限制使用的GPU个数, 如使用第0和第1编号的GPU,设置:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg['train_gpu']) 
    # =====================================================================
    # 对多 GPU 训练的输入参数进行设置
    if cfg['dist_url'] == "env://" and cfg['world_size'] == -1:
        cfg['world_size'] = int(os.environ["WORLD_SIZE"])
    cfg['distributed'] = cfg['world_size'] > 1 or cfg['multiprocessing_distributed']  # 判断是否进行 distributed 训练
    cfg['ngpus_per_node'] = len(cfg['train_gpu'])  # GPU 数量。 参数指定为当前主机创建的进程数。一般设定为当前主机的 GPU 数量。
    if len(cfg['train_gpu']) == 1:
        cfg['sync_bn'] = False
        cfg['distributed'] = False
        cfg['multiprocessing_distributed'] = False
    if cfg['multiprocessing_distributed']:
        port = find_free_port()
        cfg['dist_url'] = f"tcp://127.0.0.1:{port}"
        cfg['world_size'] = cfg['ngpus_per_node'] * cfg['world_size']   # GPU 数量*主机数。为总进程数
        # torch.multiprocessing 多进程：
        mp.spawn(main_worker, nprocs=cfg['ngpus_per_node'], args=(cfg['ngpus_per_node'], cfg))
    else:
        main_worker(cfg['train_gpu'], cfg['ngpus_per_node'], cfg)          
    
    return cfg

def main_worker(gpu, ngpus_per_node, cfgg):
    global cfg  # 声明全局变量，可在函数外使用。对 main 中的 cfg 参数进行修改？
    cfg = cfgg
    
    # ==========================================================================
    if cfg['distributed']:
        if cfg['dist_url'] == "env://" and cfg['rank'] == -1:
            cfg['rank'] = int(os.environ["RANK"])
        if cfg['multiprocessing_distributed']:
            cfg['rank'] = cfg['rank'] * ngpus_per_node + gpu   # rank 为当前进程的进程号。 # 如果 gpu =0, 用来输出、保存参数。如果 gpu !=0, 不保存参数。

        # 使用 init_process_group 初始化进程组，同时初始化 distributed 包  
        dist.init_process_group(backend=cfg['dist_backend'], init_method=cfg['dist_url'], world_size=cfg['world_size'], rank=cfg['rank'])
    # =====================================================================
    # 为何这段代码放后面不会导致 info 多次输出，而放最前面会导致多次输出:因为：cfg['rank'] = cfg['rank'] * ngpus_per_node + gpu
    if main_process():
        # 建立日志
        global logger, writer
        
        file_name_log = './save/log/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset']+'/'+"record_"+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset']  +"_train.log"

        logger = build_logger(cfg['logger_name_trainval'] , file_name_log)
        logger.info('Run =======================================================================================')

        writer = SummaryWriter('./save/tensorboardX/'+cfg['NAME_model'] +'_'+ cfg['NAME_dataset'])     # 这是可视化用的
    # ==========================================================================
    # 当使用 CrossEntropyLoss 时，函数内部先经过了 softmax 激活函数

    criterion = get_criterion(cfg)  # 加载特定的损失函数
    criterion_val = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])    # 用于计算 val loss 的 criterion

    # -----------------------------------------
    # 加载网络结构，并实例化      
    model, modules_ori, modules_new = load_model(cfg, criterion, 'train')

    # ----------------------------------
    # 获取加载的 model 中的某几层，并对这些层分别设置学习率
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=cfg['base_lr']))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=cfg['base_lr'] * 10))
    index_split = len(modules_ori)
    # 对加载的 model 层设置优化器
    optimizer = torch.optim.SGD(params_list, lr=cfg['base_lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    # 多 GPU 训练时，会影响 batch norm 的使用
    if cfg['sync_bn']:
        # convert all BatchNorm*D layers in the model to torch.nn.SyncBatchNorm layers
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)   # 使用 SyncBatchNorm

    # ---------------------------------------------------------
    if cfg['distributed']:
        # 设置当前设备。不鼓励使用此函数来设置。在大多数情况下，最好使用CUDA_VISIBLE_DEVICES 环境变量。
        torch.cuda.set_device(gpu)
        cfg['batch_size'] = int(cfg['batch_size'] / ngpus_per_node)       # 数据加到不同 GPU 上，因此 batch_size 数量需要除以 GPU 个数
        cfg['batch_size_val'] = int(cfg['batch_size_val'] / ngpus_per_node)
        cfg['num_workers'] = int((cfg['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)   # 计算加载数据集的线程数
        # model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)   # 创建分布式并行模型
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])   # 创建分布式并行模型
    else:
        model = torch.nn.DataParallel(model.cuda())

    # ---------------------------------------------------------

    if cfg['load_model_path']: # 进行网络权重初始化的地址
        if os.path.isfile(cfg['load_model_path']):   # 如果保存模型的位置有文件
            if main_process():
                logger.info("=> loading weight '{}'".format(cfg['load_model_path']))
            checkpoint = torch.load(cfg['load_model_path'])  # 加载网络
            model.load_state_dict(checkpoint['state_dict'])   # 加载网络权值
            if main_process():
                logger.info("=> loaded weight '{}'".format(cfg['load_model_path']))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(cfg['load_model_path']))

    if cfg['load_checkpoint'] and cfg['if_load_checkpoint']:
        if os.path.isfile(cfg['load_checkpoint']):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(cfg['load_checkpoint']))

            # 从记录的checkpoint进行网络训练的恢复
            checkpoint = torch.load(cfg['load_checkpoint'], map_location=lambda storage, loc: storage.cuda())
            cfg['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg['load_checkpoint'], checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(cfg['load_checkpoint']))

    # =============================================================
    # 加载数据集
    # 创建训练集和测试集的实例
    value_scale = cfg['value_scale']
    mean = cfg['mean']
    mean = [item * value_scale for item in mean]
    std = cfg['std']
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        # 随机尺度变换
        transform.RandScale([cfg['scale_min'], cfg['scale_max']]),

        # 随机旋转
        transform.RandRotate([cfg['rotate_min'], cfg['rotate_max']], padding=mean, ignore_label=cfg['ignore_label']),
        # 以概率 p 对图像进行随机高斯滤波：仅用于 VOC 数据集
        # transform.RandomGaussianBlur(),   

        # 以概率 p 进行水平翻转:利用翻转模拟原文的随机镜像
        transform.RandomHorizontalFlip(),

        # 利用 mean 对图像进行边界填充，用 ignore_label 对 label 进行边界填充
        transform.Crop([cfg['train_h'], cfg['train_w']], crop_type='rand', padding=mean, ignore_label=cfg['ignore_label']),
        # np.ndarray 转 tensor
        transform.ToTensor(),
        # 标准化
        transform.Normalize(mean=mean, std=std)])

    val_transform = transform.Compose([
        # 利用 mean 对图像进行边界填充，用 ignore_label 对 label 进行边界填充
        transform.Crop([cfg['train_h'], cfg['train_w']], crop_type='center', padding=mean, ignore_label=cfg['ignore_label']),
        # np.ndarray 转 tensor
        transform.ToTensor(),
        # 标准化
        transform.Normalize(mean=mean, std=std)])

    # 读取、转换、并将 img、label 定义为可迭代类
    data_train = dataset_loader_general.SegDataset(data_root=cfg['data_root'],
                    list_root=cfg['train_list_root'],split='train',max_num=cfg['max_num'],
                    transform=train_transform,num_classes=cfg['num_classes'],
                    cfg=cfg)

    # 为数据集创建 Sampler
    if cfg['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
    else:
        train_sampler = None

    if main_process():
        logger.info('加载的训练集大小: {}'.format(len(data_train)))

    if cfg['if_val']:
        data_val = dataset_loader_general.SegDataset(data_root=cfg['data_root'],
                    list_root=cfg['val_list_root'],split='val',max_num=cfg['max_num'],
                    transform=val_transform,num_classes=cfg['num_classes'],
                    cfg=cfg)
        
        # 为数据集创建 Sampler
        if cfg['distributed']:
            val_sampler = torch.utils.data.distributed.DistributedSampler(data_val)
        else:
            val_sampler = None
        
        if main_process():
            logger.info('加载的验证集大小: {}'.format(len(data_val)))

    # 训练数据加载到 loader 中 (train_sampler is None)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=cfg['batch_size'], shuffle=(train_sampler is None), # 洗牌
                                                num_workers=cfg['num_workers'],sampler=train_sampler, pin_memory=True,  # 子进程，复制数据到CUDA
                                                drop_last=True) # 丢弃不满 batch_size 的一组数据
    

    if cfg['if_val']:
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=cfg['batch_size_val'], shuffle=False, # 洗牌
                                                num_workers=cfg['num_workers'], sampler=val_sampler, pin_memory=True,  # 子进程，复制数据到CUDA
                                                drop_last=True) # 丢弃不满 batch_size 的一组数据

    # ====================================================================
    # 需要保存的参数进行初始化
    since = time.time()  # 计时
    best_train_acc = 0.0
    best_train_epoch = 0
    best_eval_acc = 0.0
    best_eval_epoch = 0

    if cfg['if_train']:
        if main_process():
            logger.info('---------------->开始进行训练')

        for epoch in range(cfg['start_epoch'],cfg['epochs']):
            # 进行训练====================================
            epoch_log = epoch + 1    # 检查点保存的 epoch

            if cfg['distributed']:
                train_sampler.set_epoch(epoch)

            epoch_time_elapsed, acc_class, acc_mean, IoU_class, IoU_mean, train_loss = trainer(model,train_loader,cfg,optimizer,epoch,index_split)

            if main_process():
                # 将训练结果输出
                logger.info('Epoch: {}/{}, Train Loss: {:.5f}, Train mAcc: {:.5f}, Train mIoU: {:.5f}, epoch time: {:.5f}s '.format(
                        epoch+1,cfg['epochs'], train_loss, acc_mean, IoU_mean, epoch_time_elapsed))

                # tensorboardX 可视化
                writer.add_scalar('loss_train', train_loss, epoch)
                writer.add_scalar('mIoU_train', IoU_mean, epoch)
                writer.add_scalar('mAcc_train', acc_mean, epoch)
                # writer.add_scalar('allAcc_train', allAcc_train, epoch)

                # 记录 train 精度最高的模型
                epoch_train_acc = IoU_mean
                if epoch_train_acc > best_train_acc:
                    best_train_epoch = epoch
                    best_train_acc = epoch_train_acc
                    best_train_model_dict = model.state_dict()  # 记录训练精度最高的模型参数
                    best_train_optimizer_dict = optimizer.state_dict()


            if (epoch_log % cfg['save_freq'] == 0) and main_process():
                filename = './save/model/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + str(epoch_log) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)

                # 保存模型、优化器的参数
                torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
                if epoch_log / cfg['save_freq'] > 2:
                    deletename = './save/model/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + str(epoch_log - cfg['save_freq'] * 2) + '.pth'
                    if os.path.isfile(deletename):
                        os.remove(deletename)

            if cfg['if_val']:  # 是否进行 val
                # 每隔 val_val_freq-1 个批次进行 val =============================
                if (epoch+1) % cfg['val_freq'] ==0:
                    # logger.info('---------------->开始进行 val')
                    model.eval()   # 很重要
                    # 进行 val
                    epoch_time_elapsed_val, acc_class_val, acc_mean_val, IoU_class_val, IoU_mean_val, loss_val = validater(model,val_loader,cfg,criterion_val)

                    if main_process():
                        # 将训练结果输出
                        logger.info('Epoch: {}/{}, val Loss: {:.5f}, val mAcc: {:.5f}, val mIoU: {:.5f}, val epoch time: {:.5f}s '.format(
                                epoch+1,cfg['epochs'], loss_val, acc_mean_val, IoU_mean_val, epoch_time_elapsed_val))

                        # tensorboardX 可视化
                        writer.add_scalar('loss_val', loss_val, epoch)
                        writer.add_scalar('mIoU_val', IoU_mean_val, epoch)
                        writer.add_scalar('mAcc_val', acc_mean_val, epoch)
                        # writer.add_scalar('allAcc_train', allAcc_train, epoch)
                        
                        # 记录 val 精度最高的模型
                        epoch_eval_acc = IoU_mean_val
                        if epoch_eval_acc > best_eval_acc:
                            best_eval_epoch = epoch
                            best_eval_acc = epoch_eval_acc
                            best_val_model_dict = model.state_dict()  # 记录训练精度最高的模型参数
                            best_val_optimizer_dict = optimizer.state_dict()

    if cfg['if_val']:  # 是否进行 val
        if main_process():
            logger.info('---------------->开始进行 final val')
            
        model.eval()   # 很重要
        # 进行 val   
        epoch_time_elapsed_val, acc_class_val, acc_mean_val, IoU_class_val, IoU_mean_val, loss_val = validater(model,val_loader,cfg, criterion_val)

        if main_process():
            # 将训练结果输出
            logger.info('Epoch: {}/{}, val Loss: {:.5f}, val mAcc: {:.5f}, val mIoU: {:.5f}, val epoch time: {:.5f}s '.format(
                    cfg['epochs'],cfg['epochs'], loss_val, acc_mean_val, IoU_mean_val, epoch_time_elapsed_val))

            # tensorboardX 可视化
            writer.add_scalar('loss_val', loss_val, cfg['epochs'])
            writer.add_scalar('mIoU_val', IoU_mean_val, cfg['epochs'])
            writer.add_scalar('mAcc_val', acc_mean_val, cfg['epochs'])
            # writer.add_scalar('allAcc_train', allAcc_train, epoch)
            
            # 记录 val 精度最高的模型
            epoch_eval_acc = IoU_mean_val
            if epoch_eval_acc > best_eval_acc:
                best_eval_epoch = cfg['epochs']
                best_eval_acc = epoch_eval_acc
                best_val_model_dict = model.state_dict()  # 记录训练精度最高的模型参数
                best_val_optimizer_dict = optimizer.state_dict()

    time_elapsed = time.time() - since
    if main_process():
        logger.info('Training & val complete in {:.0f}m {:.5f}s'.format(
            time_elapsed//60, time_elapsed % 60))

        if cfg['if_train']:
            # 保存 train 精度最高的模型
            save_filename = './save/model_best/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+'best_train_' + cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + '.pkl'
            torch.save({'epoch': best_train_epoch, 'state_dict': best_train_model_dict, 'optimizer': best_train_optimizer_dict}, save_filename)

            logger.info('saving dict.....: best_train_epoch= {}, file_name: {}'.format(best_train_epoch,save_filename)) 

        if cfg['if_val']:
            # 保存 val 精度最高的模型
            save_filename = './save/model_best/'+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'/'+'best_val_' +cfg['NAME_model'] +'_'+ cfg['NAME_dataset'] +'_'+ str(len(cfg['train_gpu']))+'GPU_train'+ cfg['save_model_filename'] + '.pkl'
            torch.save({'epoch': best_eval_epoch, 'state_dict': best_val_model_dict, 'optimizer': best_val_optimizer_dict}, save_filename)
            logger.info('saving dict.....: best_val_epoch= {}, file_name: {}'.format(best_eval_epoch,save_filename))


# =====================================================================
from util import read_yaml

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--yaml_Name', type=str, default='MFARANet_train_valuation_Basic_Config.yaml', help='yaml_Name')

parser.add_argument('--NAME_model', type=str, default='MFARANet_resnet_18_deep_stem', help='NAME_model')

parser.add_argument('--Marker', type=str, help='Marker')

parser.add_argument('--Branch_Choose', nargs='+', type=int, help='Branch_Choose')

parser.add_argument('--base_lr', type=float,  default=0.005, help='learning rate')

parser.add_argument('--weight_decay', type=float,  default=0.0005, help='weight_decay')

parser.add_argument('--if_train_val', required=False, default=False, action="store_true", help="if_train_val")

parser.add_argument('--load_trained_model', type=str, help='path: load_trained_model')

parser.add_argument('--Dropout_Rate_CNN', nargs='+', type=float, help='Dropout_Rate_CNN')

parser.add_argument('--backbone_name', type=str, default='resnet18_deep_stem', help='backbone_name')

parser.add_argument('--train_gpu', nargs='+', type=int, help='train_gpu')

parser.add_argument('--epochs', type=int, default=200, help='epochs')

args = parser.parse_args()

# ===================================================================================
if __name__ == '__main__':
    # 读取 yaml 文件
    yaml_path = './config/cityscapes/MFARANet/' + args.yaml_Name

    cfg = read_yaml.read_config(yaml_path)   # 读取 yaml 文件

    cfg['NAME_model'] = args.NAME_model

    if args.Marker is not None:
        cfg['NAME_model'] = args.NAME_model + '_' + args.Marker

    if args.Branch_Choose is not None:
        cfg['Branch_Choose'] = args.Branch_Choose

    cfg['base_lr'] = args.base_lr

    cfg['save_model_filename'] = '_lr_' + str(args.base_lr).split('.')[-1] + '_batch_14_200_'

    cfg['weight_decay'] = args.weight_decay

    cfg['if_train_val'] = args.if_train_val

    if args.load_trained_model is not None:
        cfg['load_trained_model'] = args.load_trained_model

    if args.Dropout_Rate_CNN is not None:
        cfg['Dropout_Rate_CNN'] = args.Dropout_Rate_CNN 

    cfg['backbone_name'] = args.backbone_name

    if args.train_gpu is not None:
        cfg['train_gpu'] = args.train_gpu

    cfg['epochs'] = args.epochs

    # ----------------------------------------------------------
    if cfg['if_train_val']:
        main(cfg)

    if cfg['if_inference']:
        main_inference(cfg)   # MS inference


    if cfg['if_demo']:     # 单张图片的 demo
        main_demo(cfg)