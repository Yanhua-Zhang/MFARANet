import time
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
import torch
import numpy as np
import torch.distributed as dist


def trainer(model,train_loader,cfg,optimizer,epoch,index_split):
    
    model.train()
    
    train_loss = 0
    intersection_sum = 0   
    union_sum = 0
    target_sum = 0  
    intersection, union, target = 0, 0, 0
    
    max_iter = cfg['epochs'] * len(train_loader)  # 计算总迭代次数：epoch*len(train)/batch_size
    count_img = 0                          # 训练样本总数

    # for data in train_data:
    epoch_since = time.time()  # 计时
    for i, data in enumerate(train_loader):
        
        im, label, binary_boundary_mask = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), \
                                            data[2].cuda(non_blocking=True)   # 2 值的 boundary mask

        # im, label = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)  # 2 值的 boundary mask
    
        # 输出 最终 loss 和 中间层 loss (辅助loss)，loss 在网络中进行计算       
        out, loss = model(im, (label, binary_boundary_mask))  # 这里的各类 mask 打包送入网络，可以使代码通用性变强。

        # 若不多线程
        if not cfg['multiprocessing_distributed']:           
            loss = torch.mean(loss)  # 所有元素平均值

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = im.size(0)
        if cfg['multiprocessing_distributed']:  # 如果多线程
            # 利用 detach() 函数来切断一些分支的反向传播
            loss = loss * n  # not considering ignore pixels

            # new_tensor()可以将源张量中的数据复制到目标张量（数据不共享）
            count = label.new_tensor([n], dtype=torch.long)
            # 将各线程数据进行 average 然后返回到 rank 0 中进行输出（用于backward 的 loss 不需要 dist.all_reduce
            dist.all_reduce(loss), dist.all_reduce(count)
            
            # item()的作用是取出单元素张量的元素值并返回该值，保持该元素类型不变。
            # 取出的元素值的精度更高，所以在求损失函数等时我们一般用item（）
            n = count.item() 
            loss = loss / n
        
        # loss.item()*im.size(0): 一个 batch_size 的总 loss。 相加后为一个 epoch 所有图片的 loss
        train_loss += loss.item()*n 
        count_img += n   # 计算用于训练的图片总数，因为一个 batch 的图片个数可能不完整，因此进行计数非常有必要  
        
        # 一个 batch 下统计值
        # 交集、并集 的计算：用于计算 mIoU
        # intersection, union 均为 1xK 的张量
        intersection, union, target = intersectionAndUnionGPU(out, label, cfg['num_classes'], cfg['ignore_label'])

        # 如果不加这句代码，计算出的 mIoU、acc 指标是错的
        if cfg['multiprocessing_distributed']:   
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()   # 转 CPU、tensor 转 np

        # 一个 epoch 下的统计值
        intersection_sum += intersection   
        union_sum += union
        target_sum += target

        # 学习率的更新
        current_iter = epoch * len(train_loader) + i + 1  # 计算当前迭代次数
        current_lr = poly_learning_rate(cfg['base_lr'], current_iter, max_iter, power=cfg['power'])  # 学习率进行更新
        for index in range(0, index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

    epoch_time_elapsed = time.time() - epoch_since      # 计时

    acc_class = intersection_sum / (target_sum + 1e-10)   # 类别精度
    acc_mean = np.mean(acc_class)                         # 类别平均精度

    IoU_class = intersection_sum / (union_sum + 1e-10)   # 类别精度
    IoU_mean = np.mean(IoU_class)                         # 类别平均精度

    train_loss = train_loss / count_img
    return epoch_time_elapsed, acc_class, acc_mean, IoU_class, IoU_mean, train_loss