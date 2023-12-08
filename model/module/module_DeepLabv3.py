import torch
from torch import nn
import torch.nn.functional as F

class ASPPv3_head(nn.Module):   # 继承 nn.Module 这个类
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPPv3_head, self).__init__()    # 继承父类
        self.features = []

        # aspp 中，3x3 的 conv
        for rate in atrous_rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate,
                  dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True))
            )      
        # aspp 中，1x1 的 conv
        self.features.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, padding=0,
                  dilation=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True))
            )
        # 容器：nn.ModuleList 和 nn.Sequential 都作为容器使用，但有所不同
        self.features = nn.ModuleList(self.features)

        self.avg = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                                    # 输出特征值大小为 （bin,bin)
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),    # 利用 1x1 卷积核进行 channel 调整
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x_size = x.size()  # 输入 feature map 的尺寸
        out = []
        for f in self.features:
            # 由于进行了 avg pool 输出维度减小，因此需要将各输出进行差值，调整为同一尺寸
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        img_level_feature = F.interpolate(self.avg(x), x_size[2:], mode='bilinear', align_corners=True)
        out.append(img_level_feature)

        return torch.cat(out, 1)   # 沿 channel 维度进行拼接