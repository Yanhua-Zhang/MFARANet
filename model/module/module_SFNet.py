import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential

class FAM_module(nn.Module):
    def __init__(self, in_channel_low_feature, in_channel_h_feature, middle_channel1, middle_channel2):
        super(FAM_module, self).__init__()

        # 一个 1x1 conv 将 low_feature channel 统一调整为 middle_channel1（256）
        self.conv1x1_input = nn.Sequential(
            nn.Conv2d(in_channel_low_feature, middle_channel1, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channel1),
            nn.ReLU(inplace=True),
        )  

        self.conv1x1_1 = nn.Conv2d(middle_channel1, middle_channel2, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整    
        self.conv1x1_2 = nn.Conv2d(in_channel_h_feature, middle_channel2, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整     
        self.conv3x3_1 = nn.Conv2d(middle_channel2*2, 2, kernel_size=3, padding=1, bias=False)    # 用于计算 offset map

           
    def forward(self, low_feature, h_feature):
        n, c, h, w = low_feature.size()           # low_feature 的 size
        h_feature_origin = h_feature              #
        h_feature = self.conv1x1_2(h_feature)      # 小特征层进行 1x1 conv
        h_feature = F.interpolate(h_feature, (h, w), mode='bilinear', align_corners=True)   # 插值到大特征层的大小

        low_feature = self.conv1x1_input(low_feature)
        out_low_feature = low_feature               # 将进行 channel 调整后的 low_feature 输出，用于 skip connect
        low_feature = self.conv1x1_1(low_feature)    # 再次将 channel 进行调整，用于计算 offset

        fuse_feature = torch.cat([low_feature,h_feature], 1) # 融合
        flow_field = self.conv3x3_1(fuse_feature)       # 计算 Flow Field：[n,2,h,w]

        norm = torch.tensor([[[[w,h]]]]).type_as(low_feature).to(low_feature.device)   # 用于归一化，grid 值要在 -1~1 之间：[1,1,1,2]
        grid_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)              # [h,w]
        grid_w = torch.linspace(-1,1,w).repeat(h,1)                         # [h,w]
        grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), 2)                           # 生成用于 grid_upsample 的网格：[h,w,2]  
        grid = grid.repeat(n,1,1,1).type_as(low_feature).to(low_feature.device)    # [n,h,w,2]

        warp_grid = grid + flow_field.permute(0,2,3,1)/norm      # 论文 Eq(2) 要除以 2，但论文代码并没有除以 2

        out_h_feature = F.grid_sample(h_feature_origin, warp_grid)  # 利用网格法进行 feature 的 upsample
        # output = self.conv3x3_output(output)

        return out_low_feature + out_h_feature   # 进行 skip connect 后返回