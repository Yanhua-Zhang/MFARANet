import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['PPM_HMSA']

class PPM_HMSA(nn.Module):   # 继承 nn.Module 这个类
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM_HMSA, self).__init__()    # 继承父类
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                # 自适应均值池化：直接传入输出大小，自适应选择滤波器大小和步长，
                nn.AdaptiveAvgPool2d(bin),                                    # 输出特征值大小为 （bin,bin)
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),    # 利用 1x1 卷积核进行 channel 调整
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        # 容器：nn.ModuleList 和 nn.Sequential 都作为容器使用，但有所不同
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()  # 输入特征的尺寸
        out = []
        for f in self.features:
            # 由于进行了 avg pool 输出维度减小，因此需要将各输出进行差值，调整为同一尺寸
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out.append(x)

        return out   # 输出的特征 map 的 scale 由小到大排列，但都被 upsample 到同一维度