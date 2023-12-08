import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['SegmentationLosses', 'SegmentationMultiLosses', 'LabelSmoothing', 'NLLMultiLabelSmooth', 
            'JointEdgeSegLoss', 'BalanceCrossEntropyLoss2d', 'JointEdgeSegLoss_OHEM',
            'OhemCrossEntropy2dTensor']

# ===========================================================
# 根据一个图像或一个 batch 中像素点的分布计算 balanced loss
# 与 GSCNN、InverseForm 中的 ImageBasedCrossEntropyLoss2d 完全相同，只是换了个名字

# 根据一个 batch/单张 图像的像素点分布，计算 balance CrossEntropyLoss
class BalanceCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(BalanceCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction='mean',
                                   ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(target, bins=self.num_classes, min=0.0,
                           max=self.num_classes)   # 这里 max 不应该是 bins-1 吗？？？
        hist_norm = bins.float() / bins.sum()
        
        # 对出现过的类别，根据 bins 计算权重
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets, do_rmi=None):
        
        # 根据一整个 batch 中 mask 的像素点的类别分布计算 weight
        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0

        # 对每一个 batch 依次计算 loss,然后相加
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets[i])  # 根据单张 mask 的像素点的类别分布计算 weight
                # if self.fp16:
                #     weights = weights.half()
                self.nll_loss.weight = weights

            # 由于 inputs[i] 后，tensor 会自动减少一个维度，因此需要用 unsqueeze 再添加一个维度
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0),)
        return loss

# ===========================================================
# GSCNN、InverseForm 中的 loss

# 根据一个 batch/单张 图像的像素点分布，计算 balance CrossEntropyLoss
class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction='mean',
                                   ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(target, bins=self.num_classes, min=0.0,
                           max=self.num_classes)   # 这里 max 不应该是 bins-1 吗？？？
        hist_norm = bins.float() / bins.sum()
        
        # 对出现过的类别，根据 bins 计算权重
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets, do_rmi=None):
        
        # 根据一整个 batch 中 mask 的像素点的类别分布计算 weight
        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0

        # 对每一个 batch 依次计算 loss,然后相加
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets[i])  # 根据单张 mask 的像素点的类别分布计算 weight
                # if self.fp16:
                #     weights = weights.half()
                self.nll_loss.weight = weights

            # 由于 inputs[i] 后，tensor 会自动减少一个维度，因此需要用 unsqueeze 再添加一个维度
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0),)
        return loss


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0,
                 edge_weight=0.3, inv_weight=0.3, seg_weight=1, att_weight=0.1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, weight=weight, ignore_index=ignore_index, norm=norm, upper_bound=upper_bound).cuda()
        # self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight

    # 计算 balance 2 值 loss
    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, 
                             torch.where(edge.max(1)[0] > 0.8, target, filler))  # 仅对 edge scor map 中被判断为 boundary 的区域计算 loss

    def forward(self, inputs, targets, do_rmi=None):
        #self.inverse_distance.inversenet.zero_grad()
        segin, edgein = inputs        # edgein 应该是一个 1 channel 的 score map
        segmask, edgemask = targets
        # edgemask = edgemask.cuda()

        losses = {}
        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask, do_rmi)
        losses['edge_loss'] = self.edge_weight * self.bce2d(edgein, edgemask)
        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein) # 仅对 edge scor map 中被判断为 boundary 的区域计算 loss

        # losses['inverse_loss'] = self.inv_weight * self.inverse_distance(edgein, edgemask)
        
        total_loss = losses['seg_loss'] + losses['edge_loss'] + losses['att_loss'] 
                    # + losses['inverse_loss']  

        return total_loss

# ===========================================================
# GSCNN、InverseForm 中的 loss 加入 OHEM

# 根据一个 batch/单张 图像的像素点分布，计算 balance CrossEntropyLoss
class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)   # 判断 target 中的值是否不等于 ignore_index。不等于为 true，等于为 false
        target = target * valid_mask.long()        # 有效像素*1，无效像素乘 0
        num_valid = valid_mask.sum()             # 总的有效像素点个数

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class JointEdgeSegLoss_OHEM(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0,
                 thresh=0.7, min_kept=100000,
                 edge_weight=0.3, inv_weight=0.3, seg_weight=1, att_weight=0.1, edge='none'):
        super(JointEdgeSegLoss_OHEM, self).__init__()
        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, weight=weight, ignore_index=ignore_index, norm=norm, upper_bound=upper_bound).cuda()
                
        self.Ohem_loss = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept).cuda()
        # self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight

    # 计算 balance 2 值 loss
    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, 
                             torch.where(edge.max(1)[0] > 0.8, target, filler))  # 仅对 edge scor map 中被判断为 boundary 的区域计算 loss

    def forward(self, inputs, targets, do_rmi=None):
        #self.inverse_distance.inversenet.zero_grad()
        segin, edgein = inputs        # edgein 应该是一个 1 channel 的 score map
        segmask, edgemask = targets
        # edgemask = edgemask.cuda()

        losses = {}
        losses['seg_loss'] = self.seg_weight * self.Ohem_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * self.bce2d(edgein, edgemask)
        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein) # 仅对 edge scor map 中被判断为 boundary 的区域计算 loss

        # losses['inverse_loss'] = self.inv_weight * self.inverse_distance(edgein, edgemask)
        
        total_loss = losses['seg_loss'] + losses['edge_loss'] + losses['att_loss'] 
                    # + losses['inverse_loss']  

        return total_loss

# ============================================================
# BFPNet 中使用的 loss, 注意输入为 2个 pred 和 2个 target
class SegmentationMultiLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None, size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass

    def forward(self, *inputs):

        pred1, pred2, target, target2 = tuple(inputs)

        # pred1, pred2 = tuple(preds)

        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)   # 这里应该用的是 nn.CrossENtropyLoss 的 forward
        loss2 = super(SegmentationMultiLosses, self).forward(pred2, target2)

        loss = loss1 + loss2
        return loss

# ============================================================
# EncNet 提出的 se loss 和最常用的 aux loss
class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:  # aux loss
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:      # se loss
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:                   # aux loss + se loss ？？？
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)  # main loss
            loss2 = super(SegmentationLosses, self).forward(pred2, target)  # aux loss
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)         # se loss
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

# ============================================================================
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

# ===================================================================================
