import time

import torch
from torch import nn
import torch.nn.functional as F

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs
# --------------------------------------------------------------------------------
# 检查保存 summary log 的文件夹是否存在，不存在进行生成
summary_path = '/home/zhangyanhua/Code_python/Semantic-seg-multiprocessing-general-Test/save/summary'
check_makedirs(summary_path)            

file_name_log = summary_path+'/'+"model_module_time.log"  # logger 的文件名
logger = build_logger('model_module_time', file_name_log)
# ----------------------------------------------------------

channel = 128
height = 250
weight = 250
# input = torch.Tensor(1, channel, height, weight).cuda()
input = torch.Tensor(1, channel, height, weight)

logger.info("start warm up")
for i in range(10):
    x_avg = F.avg_pool2d(input, input.size()[2:])
logger.info("warm up done")

# ---------------------------------------------------------
logger.info('第一个操作')
start_ts = time.time()
for i in range(100):
    x_avg = F.avg_pool2d(input, input.size()[2:])     # 1
# torch.cuda.synchronize() 
end_ts = time.time()
t_cnt = end_ts - start_ts

logger.info("FPS: %f" % (100 / t_cnt))
logger.info(f"Inference time {t_cnt/100*1000} ms")
logger.info('End')
logger.info('--------------------------------')
logger.info('        ')

# ---------------------------------------------------------
attention =nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.Sigmoid(),
                )
attention.eval()

logger.info('第二个操作')
start_ts = time.time()
for i in range(100):
    channel_attention = attention(x_avg)          # 2
# torch.cuda.synchronize() 
end_ts = time.time()
t_cnt = end_ts - start_ts

logger.info("FPS: %f" % (100 / t_cnt))
logger.info(f"Inference time {t_cnt/100*1000} ms")
logger.info('End')
logger.info('--------------------------------')
logger.info('        ')

# ---------------------------------------------------------
logger.info('第3个操作')
start_ts = time.time()
for i in range(100):
    enhance_feat = torch.mul(input, channel_attention)    # 3
# torch.cuda.synchronize() 
end_ts = time.time()
t_cnt = end_ts - start_ts

logger.info("FPS: %f" % (100 / t_cnt))
logger.info(f"Inference time {t_cnt/100*1000} ms")
logger.info('End')
logger.info('--------------------------------')
logger.info('        ')

# ---------------------------------------------------------
logger.info('第4个操作')
start_ts = time.time()
for i in range(100):
    out = input + enhance_feat
# torch.cuda.synchronize() 
end_ts = time.time()
t_cnt = end_ts - start_ts

logger.info("FPS: %f" % (100 / t_cnt))
logger.info(f"Inference time {t_cnt/100*1000} ms")   # 4
logger.info('End')
logger.info('--------------------------------')
logger.info('        ')

# enhance_feat = torch.mul(x, channel_attention)    

# x = x + enhance_feat   # skip connect