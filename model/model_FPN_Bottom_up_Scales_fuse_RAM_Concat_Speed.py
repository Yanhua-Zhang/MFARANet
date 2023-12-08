import torch
from torch import nn
import torch.nn.functional as F

from .backbone import resnet18_deep_stem, resnet50_deep_stem   # 加载预训练模型与参数
from .module import FAM_module, PPM

# offset map 的计算
class SFNet_warp_grid(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(SFNet_warp_grid, self).__init__()

        self.channel_change1 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整    
        self.channel_change2 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)    # 利用 1x1 卷积核进行 channel 调整 
        self.offset_map = nn.Conv2d(middle_channel*2, 2, kernel_size=3, padding=1, bias=False)    # 用于计算 offset map
           
    def forward(self, low_feature, h_feature):
        n, c, h, w = low_feature.size()           # low_feature 的 size
        
        h_feature = self.channel_change1(h_feature)   # channel 调整
        h_feature_up = F.interpolate(h_feature, (h, w), mode='bilinear', align_corners=True)   # 插值到大特征层的大小
        low_feature = self.channel_change2(low_feature)   # channel 调整

        fuse_feature = torch.cat([low_feature, h_feature_up], 1)         # 融合
        flow_field = self.offset_map(fuse_feature)                       # 计算 Flow Field：[n,2,h,w]

        norm = torch.tensor([[[[w,h]]]]).type_as(low_feature).to(low_feature.device)   # 用于归一化，grid 值要在 -1~1 之间：[1,1,1,2]
        grid_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)              # [h,w]
        grid_w = torch.linspace(-1,1,w).repeat(h,1)                         # [h,w]
        grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), 2)                           # 生成用于 grid_upsample 的网格：[h,w,2]  
        grid = grid.repeat(n,1,1,1).type_as(low_feature).to(low_feature.device)    # [n,h,w,2]

        warp_grid = grid + flow_field.permute(0,2,3,1)/norm      # 论文 Eq(2) 要除以 2，但论文代码并没有除以 2

        # out_h_feature = F.grid_sample(h_feature_origin, warp_grid)  # 利用网格法进行 feature 的 upsample
        return warp_grid


class FPN_Bottom_up_Scales_fuse(nn.Module):
    def __init__(self, backbone_name='resnet18_deep_stem', use_aux_loss=True, 
                    use_dilation=False, use_PPM=False, ppm_bins=(1, 2, 3, 6), 
                    if_stage1_4_repeat_fuse=False, HMSA_stage_choose =(1,2,3,4),
                    if_use_boundary_loss = False,
                    fam_dim=128, aux_weight= 0.4,  classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(FPN_Bottom_up_Scales_fuse, self).__init__()

        self.criterion = criterion
        
        self.use_dilation = use_dilation  
        self.use_PPM = use_PPM

        self.aux_weight = aux_weight
        self.use_aux_loss = use_aux_loss
        # self.use_Multi_loss = use_Multi_loss
        self.if_stage1_4_repeat_fuse = if_stage1_4_repeat_fuse
        self.HMSA_stage_choose = HMSA_stage_choose
        # -------------------------------------------------------------------
        # backbone 加载
        if backbone_name == 'resnet18_deep_stem':
            resnet = resnet18_deep_stem(pretrained=True)
            stage_channels = [64, 128, 256, 512]  # 'resnet18' 各 stage 的输出 channel
        elif backbone_name == 'resnet50_deep_stem':
            resnet = resnet50_deep_stem(pretrained=True)
            stage_channels = [256, 512, 1024, 2048]  # 'resnet18' 各 stage 的输出 channel
        
        # 利用 imagenet 预训练层构建 backbone
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 
        del resnet  # 这里删除变量名，释放内存
        
        # -------------------------------------------------------------------
        # Backbone 中 layer3,layer4 的 4 个 conv 层替换为空洞卷积
        if self.use_dilation:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)

        if self.use_PPM:
            # --------------------------------------------------------------------
            # ppm 模块。注意这里 ppm 中的 channel 设置与 SFNet 原代码不同，这里进行更改后，保持与 SFNet 一致。
            self.ppm = PPM(stage_channels[3], fam_dim, ppm_bins)
            # 输出 ppm 的特征图进行 channel 调整。这里与 SFNet 中的一样。
            self.bottleneck = nn.Sequential(
                nn.Conv2d(stage_channels[3] + fam_dim*len(ppm_bins), fam_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                # nn.BatchNorm2d(fam_dim),
                nn.ReLU(),
                nn.Dropout2d(0.1)
                )

        # --------------------------------------------------------------------
        # 各 stage 的 feature 统一进行 channel 的调整
        self.channel_changes = []   # 顺序： [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
        for stage_channel in stage_channels:
            # 进行各 stage 的 channel 调整
            self.channel_changes.append(nn.Sequential(
                nn.Conv2d(stage_channel, fam_dim, kernel_size=1, bias=False),
                # nn.BatchNorm2d(fam_dim),
                nn.ReLU(inplace=True),
                ))

        self.channel_changes = nn.ModuleList(self.channel_changes) 
        
        # --------------------------------------------------------------------
        # up_branch:
        # 类似FPN。3x3 conv 进行 fuse
        self.feature_fuses_up = []     # 顺序： [stage1_feature, stage2_feature, stage3_feature]
        for stage_channel in stage_channels[:-1]:
            # 不同 stage 的 feature 进行 sum + skip connect 后，利用 3x3 conv 进行 fuse
            self.feature_fuses_up.append(nn.Sequential(
                nn.Conv2d(fam_dim, fam_dim, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(fam_dim),
                nn.ReLU(inplace=True),
            )) 
        self.feature_fuses_up = nn.ModuleList(self.feature_fuses_up)

        # --------------------------------------------------------------------
        # down_branch:
        # 反向 FPN 结构的构建。 3x3 conv 进行 fuse
        self.feature_fuses_down = []     # 顺序： [stage2_feature, stage3_feature, stage4_feature]
        for stage_channel in stage_channels[1:]:
            # 不同 stage 的 feature 进行 sum + skip connect 后，利用 3x3 conv 进行 fuse
            self.feature_fuses_down.append(nn.Sequential(
                nn.Conv2d(fam_dim, fam_dim, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(fam_dim),
                nn.ReLU(inplace=True),
            )) 
        self.feature_fuses_down = nn.ModuleList(self.feature_fuses_down)
        
        # ------------------------------------------------------------------------
        # down 和 up branch 的 feature 融合
        if self.if_stage1_4_repeat_fuse:
            self.stage_fuses = []
            for i in range(len(stage_channels)):
                self.stage_fuses.append(nn.Sequential(
                    nn.Conv2d(fam_dim, fam_dim, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(fam_dim),
                    nn.ReLU(inplace=True),
                ))
            self.stage_fuses = nn.ModuleList(self.stage_fuses)
        else:
            # 仅进行 stage2、3 阶段的 feature fuse 
            self.stage_fuses = []
            for i in range(len(stage_channels)-2):
                self.stage_fuses.append(nn.Sequential(
                    nn.Conv2d(fam_dim, fam_dim, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(fam_dim),
                    nn.ReLU(inplace=True),
                ))
            self.stage_fuses = nn.ModuleList(self.stage_fuses)            

        # ------------------------------------------------------------------------
        # 参照 SFNet 中的模块儿计算 offset map
        self.stages_offset = []
        # stage 1 与 stage 2、stage 2与 stage 3、stage 3 与 stage 4 间的 offset maps
        for i in range(len(stage_channels)-1):   
                self.stages_offset.append(SFNet_warp_grid(fam_dim, fam_dim//2))
        self.stages_offset = nn.ModuleList(self.stages_offset)

        # ------------------------------------------------------------------------
        # final loss head
        # 各 stage 的 feature 串联后进行融合，该 conv 层用于各 stage feature concat 融合后的 score map 计算。
        self.score_head = nn.Sequential(
            nn.Conv2d(fam_dim*len(stage_channels), fam_dim, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(fam_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fam_dim, classes, kernel_size=1),
        )

        # ------------------------------------------------------------------------
        if self.training:
            # ------------------------------------------------------------------------
            # 这里是计算各 fuse_stage 输出的 mulit-loss
            self.multi_loss_head = []    # 顺序： [stage1_fuse_feature, stage2_fuse_feature, stage3_fuse_feature, stage4_fuse_feature]
            for i in range(len(stage_channels)):
                self.multi_loss_head.append(
                        nn.Sequential(
                            nn.Conv2d(fam_dim, classes, kernel_size=1, stride=1, padding=0, bias=True)
                        )
                    )

            self.multi_loss_head = nn.ModuleList(self.multi_loss_head)

            # ------------------------------------------------------------------------
            # aux loss head
            if self.use_aux_loss:
                self.aux_head = nn.Sequential(
                    nn.Conv2d(stage_channels[2], int(stage_channels[2]/2), kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(int(stage_channels[2]/2)),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(int(stage_channels[2]/2), classes, kernel_size=1)
                )


    def forward(self, x, y=None):
        x_size = x.size()
        
        out_in = x
        x = self.layer0(x)
        out0 = x
        x = self.layer1(x)
        out1 = x
        stage1_feature = x
        x = self.layer2(x)
        out2 = x
        stage2_feature = x
        x = self.layer3(x)  # 用于计算中间层 loss
        out3 = x
        stage3_feature = x
        x = self.layer4(x)
        out4 = x
        stage4_feature = x
        # --------------------------------------------------------------
        # 使用 PPM 或不使用
        if self.use_PPM:
            # ppm_out = self.ppm(stage4_feature)
            # backbone_out = self.bottleneck(ppm_out)
            print('暂不考虑使用 PPM')
        # else:
        #     backbone_out = stage4_feature
        
        # --------------------------------------------------------------
        # 对各 stage 的 feature 进行 channel 调整
        stage_features = [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
        compress_stage_features = []
        for i in range(len(stage_features)):
            compress_stage_features.append(self.channel_changes[i](stage_features[i]))

        # --------------------------------------------------------------
        # 输入 Top-down branch 结构（FPN 结构）
        stage_features_up = [compress_stage_features[0], compress_stage_features[1], compress_stage_features[2]]
        f = compress_stage_features[3]                            # stage4 经过 channel 调整的特征
        FPN_features_up = [f]    
        for i in reversed(range(len(stage_features_up))):
            stage_feature = stage_features_up[i]
            f = F.interpolate(f, (stage_feature.size())[2:], mode='bilinear', align_corners=True)   # 上采样
            f = self.feature_fuses_up[i](f + stage_feature)       # skip connect 后进行 feature fuse。 
            FPN_features_up.append(f)                             # 顺序：[stage4, stage3, stage2, stage1]

        # --------------------------------------------------------------
        # 输入 Bottom-up branch 结构
        stage_features_down = [compress_stage_features[1], compress_stage_features[2], compress_stage_features[3]]
        f = compress_stage_features[0]                              # stage1 经过 channel 调整的特征
        FPN_features_down = [f]    
        for i in range(len(stage_features_down)):
            stage_feature = stage_features_down[i]  
            f = F.interpolate(f, (stage_feature.size())[2:], mode='bilinear', align_corners=True)   # 下采样
            f = self.feature_fuses_down[i](f + stage_feature)       # skip connect 后进行 feature fuse。 
            FPN_features_down.append(f)                             # 顺序：[stage1, stage2, stage3, stage4]

        # ---------------------------------------------------------------
        # 进行 Bottom-up 与 Top-down branch 的 feature fuse
        FPN_features_up.reverse() # 顺序：[stage1, stage2, stage3, stage4]
        if self.if_stage1_4_repeat_fuse:
            fuse_features = []
            for i in range(len(FPN_features_up)):
                fuse_features.append(self.stage_fuses[i](FPN_features_up[i]+FPN_features_down[i]))
        else:
            # 仅对 stage2、3 进行 feature fuse
            fuse_features = [FPN_features_up[0]]           # up 中的 stage1 
            j = 0
            for i in range(1, len(FPN_features_up)-1):
                fuse_features.append(self.stage_fuses[j](FPN_features_up[i]+FPN_features_down[i]))
                j += 1
            fuse_features.append(FPN_features_down[3])     # down 中的 stage4

        # ----------------------------------------------------------------
        # 计算 stage 1 与 stage 2、stage 2 与 stage 3、stage 3 与 stage 4 间的 offset maps
        stages_warp_grid = []
        for i in range(len(fuse_features)-1):
            stages_warp_grid.append(self.stages_offset[i](fuse_features[i], fuse_features[i+1]))

        # 利用 offset maps 对各阶段的 fuse_features 进行上采样。渐进式上采样。
        for i in self.HMSA_stage_choose:
            if i != 1:  # 上采样到 stage1 的大小
                for k in reversed(range(i-1)):
                    fuse_features[i-1] = F.grid_sample(fuse_features[i-1], stages_warp_grid[k], align_corners=False)  # 利用网格法进行 scor map 的 upsample

        # -----------------------------------------------------------------
        # 将上采样后的 fuse_features 进行 concat 然后输出
        # fuse_features   # 顺序：[stage1, stage2, stage3, stage4]
        
        fusion_out = torch.cat(fuse_features, 1)
        score_map  = self.score_head(fusion_out)
       
        out = F.interpolate(score_map, x_size[2:], mode='bilinear', align_corners=True)

        # -----------------------------------------------------------------
        # 这个是判断是否用中间层 loss 进行辅助训练
        if self.training:
            segmask = y[0]  # 输入的各类 mask 解包
            loss_Entropy = nn.CrossEntropyLoss(ignore_index=255)   # 用于计算 main_loss 和 aux_loss

            main_loss = loss_Entropy(out, segmask)
            loss = main_loss
            # ----------------------------------------------------
            # aux loss
            if self.use_aux_loss:
                # Aux loss + main loss 的计算
                aux = self.aux_head(stage3_feature)      
                aux = F.interpolate(aux, x_size[2:], mode='bilinear', align_corners=True)
                aux_loss = loss_Entropy(aux, segmask) 
                loss  = loss + self.aux_weight*aux_loss
            # ----------------------------------------------------
            # multi-stage loss 的计算
            for i in range(len(fuse_features)):   # 顺序：[stage1, stage2, stage3，stage4]
                    stage_pred_out = self.multi_loss_head[i](fuse_features[i])
                    stage_pred_out = F.interpolate(stage_pred_out, x_size[2:], mode='bilinear', align_corners=True)
                    stage_loss = self.criterion(stage_pred_out, segmask) # 各 stage loss 的输出
                    loss += stage_loss

            return out.max(1)[1], loss
        else:
            return out


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    