# ------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

# from torchsummary import summary
from torchsummary.torchsummary import summary
from torchstat import stat

from util.model_FLOPS_HRNet import get_model_summary     # HRNet 中计算方式
from util.model_FLOPS import get_model_complexity_info  # SFNet 中计算方式

from util.model_FPS import speed_test, FPS_counter

# ------------------------------------------------------------------------------------
from model.backbone import resnet18_deep_stem, resnet50_deep_stem   # 加载预训练模型与参数


# ---------------------------------------------------------------------------------------
def summary_backbone(summary_path, model_name, model):
    # --------------------------------------------          
    file_name_log = summary_path+'/'+"backbone_summary.log"  # logger 的文件名
    logger = build_logger('summary_backbone', file_name_log)


    logger.info('backbone name：'+model_name+': ')
    logger.info(model)
    logger.info('-----------------------------------------------------------------------------')
    logger.info('   ')

# --------------------------------------------------------------------------------------
# 计算 GFLOPs、params，然后存入 logger 中
def GFLOPs_params_counter(model, model_name, height, weight, logger):

    logger.info('-----------------------------------------------------------------------')
    logger.info('model name：'+model_name+': ')
    # HRNet 计算方式----------------------------
    logger.info('开始利用 HRNet 中方式进行计算：')
    dump_input = torch.rand((1, 3, height, weight))
    logger.info(get_model_summary(model, dump_input))
    logger.info('End')
    logger.info('--------------------------------')


# ------------------------------------------------------------------------------------
# 检查保存 summary log 的文件夹是否存在，不存在进行生成
summary_path = './save/summary'
check_makedirs(summary_path)            


#-------------------------------------------------------------------------------------------------
# from model.model_MFARANet_Scale_Choice import MFARANet_Scale_Choice

# ASFM_stage_choose =[1,2,3,4]
# # ASFM_stage_choose =[1]
# # ASFM_stage_choose =[2]
# # ASFM_stage_choose =[3]
# # ASFM_stage_choose =[4]
# # ASFM_stage_choose =[2,3,4]
# # ASFM_stage_choose =[1,3,4]
# # ASFM_stage_choose =[1,2,4]
# # ASFM_stage_choose =[1,2,3]

# model_name = 'MFARANet_Scale_Choice：use dilation、not use PPM' + str(ASFM_stage_choose)
# model = MFARANet_Scale_Choice(backbone_name='resnet18_deep_stem', classes=19, 
#                                 use_dilation=True, use_PPM=False, 
#                                 use_aux_loss=False, 
#                                 if_use_boundary_loss = False,
#                                 ASFM_stage_choose = ASFM_stage_choose,
#                                 pretrained=True)

# summary_backbone(summary_path, model_name, model)

# height, weight = 1024, 1024  # 1024, 1024  1024, 2048
# file_name_log = summary_path+'/'+"model_GFLOPs_params_FPS.log"  # logger 的文件名
# logger = build_logger('summary_FPS', file_name_log)

# GFLOPs_params_counter(model, model_name, height, weight, logger)
# FPS_counter(model, model_name, height, weight, logger, iteration=100)


#-------------------------------------------------------------------------------------------------
from model.model_MFARANet_Speed import MFARANet

model_name = 'MFARANet：use dilation、not use PPM'
model = MFARANet(backbone_name='resnet18_deep_stem', use_aux_loss=False, 
                        use_dilation=True, use_PPM=False, 
                        if_stage1_4_repeat_fuse=False,
                        HMSA_stage_choose =(1,2,3,4),
                        classes=19, pretrained=True)

summary_backbone(summary_path, model_name, model)

height, weight = 1024, 1024  # 1024, 1024  1024, 2048  512, 1024  768, 1536
file_name_log = summary_path+'/'+"model_GFLOPs_params_FPS.log"  # logger 的文件名
logger = build_logger('summary_FPS', file_name_log)

GFLOPs_params_counter(model, model_name, height, weight, logger)
FPS_counter(model, model_name, height, weight, logger, iteration=100)
