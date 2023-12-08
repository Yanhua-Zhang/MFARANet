import numpy as np
import time
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
import torch
import numpy as np
import torch.distributed as dist

def validater(model,val_loader,cfg,criterion):
    
    eval_loss = 0
    intersection_sum = 0   
    union_sum = 0
    target_sum = 0
    
    intersection, union, target = 0, 0, 0

    # model.eval()   # 很重要
    count_img = 0

    # for data in train_data:
    epoch_since = time.time()  # 计时
    
    # for data in val_loader:
    for i, data in enumerate(val_loader):

        im, label = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)   # 图片、标签放入 CUDA
        
        out = model(im)
        loss = criterion(out, label)
        
        n = im.size(0)
        if cfg['multiprocessing_distributed']:
            loss = loss * n  # not considering ignore pixels
            count = label.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        eval_loss += loss.item()*im.size(0)
        count_img += im.size(0)

        out = out.max(dim=1)[1]
        
        # 一个 batch 下统计值
        # 交集、并集 的计算：用于计算 mIoU
        # intersection, union 均为 1xK 的张量
        intersection, union, target = intersectionAndUnionGPU(out, label, cfg['num_classes'], cfg['ignore_label'])
        if cfg['multiprocessing_distributed']:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()   # 转 CPU、tensor 转 np

        # 一个 epoch 下的统计值
        intersection_sum += intersection   
        union_sum += union
        target_sum += target

    epoch_time_elapsed = time.time() - epoch_since      # 计时

    acc_class = intersection_sum / (target_sum + 1e-10)   # 类别精度
    acc_mean = np.mean(acc_class)                         # 类别平均精度

    IoU_class = intersection_sum / (union_sum + 1e-10)   # 类别精度
    IoU_mean = np.mean(IoU_class)                         # 类别平均精度

    eval_loss = eval_loss / count_img

    return epoch_time_elapsed, acc_class, acc_mean, IoU_class, IoU_mean, eval_loss