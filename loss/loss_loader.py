from .loss import *
from torch import nn

def get_criterion(cfg):
    if cfg['loss_name'] == 'BFPNet_loss':
        criterion = SegmentationMultiLosses(nclass=cfg['num_classes'], ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'se_loss+aux_loss':
        criterion = SegmentationLosses(se_loss=True, se_weight=cfg['se_weight'], nclass=cfg['num_classes'],
                aux=True, aux_weight=cfg['aux_weight'], weight=None, ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'JointEdgeSegLoss':
        criterion = JointEdgeSegLoss(classes=cfg['num_classes'], ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'BalanceCrossEntropyLoss2d':
        criterion = BalanceCrossEntropyLoss2d(classes=cfg['num_classes']+1, ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'JointEdgeSegLossOHEM':
        criterion = JointEdgeSegLoss_OHEM(classes=cfg['num_classes'], ignore_index=cfg['ignore_label'])
    elif cfg['loss_name'] == 'OHEM':
        criterion = OhemCrossEntropy2dTensor(ignore_index=255, thresh=0.7, min_kept=100000, use_weight=False)

    return criterion