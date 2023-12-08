import torch
from torch import nn
import torch.nn.functional as F

from .backbone import resnet18_deep_stem, resnet50_deep_stem   # 加载预训练模型与参数
from .module import FAM_module, PPM


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
        self.if_use_boundary_loss = if_use_boundary_loss   # 是否使用 boundary loss
        
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
        # 这里是计算各 fuse_stage 输出的 mulit-loss
        # if self.use_Multi_loss:
        self.multi_loss_head = []    # 顺序： [stage1_fuse_feature, stage2_fuse_feature, stage3_fuse_feature, stage4_fuse_feature]

        for i in range(len(stage_channels)):
            self.multi_loss_head.append(
                    nn.Sequential(
                        # nn.Conv2d(fam_dim, fam_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(fam_dim),
                        # nn.ReLU(),
                        # nn.Dropout2d(0.1),
                        nn.Conv2d(fam_dim, classes, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.multi_loss_head = nn.ModuleList(self.multi_loss_head)

        # ------------------------------------------------------------------------
        # 用于计算 spatial-attention 模块
        self.HMSA_attentions = []
        # 只需要对选择的 stage 计算 attention head
        for i in range(len(HMSA_stage_choose)):
            self.HMSA_attentions.append(nn.Sequential(
                # nn.Conv2d(fam_dim, int(fam_dim/2), kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(int(fam_dim/2)),
                # nn.ReLU(inplace=True),
                nn.Conv2d(fam_dim, int(fam_dim/2), kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(int(fam_dim/2)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(int(fam_dim/2), classes, kernel_size=1, bias=False),
                nn.Sigmoid()
                )
                )
        self.HMSA_attentions = nn.ModuleList(self.HMSA_attentions)

        # ------------------------------------------------------------------------
        if self.training:
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
            # -------------------------------------------------------------------------
            # boundary loss
            if self.if_use_boundary_loss:
                self.boundary_heads = []
                for i in range(len(stage_channels)):
                    self.boundary_heads.append(nn.Sequential(
                        nn.Conv2d(fam_dim, 1, kernel_size=1, stride=1, padding=0,
                        bias=True)))
                self.boundary_heads = nn.ModuleList(self.boundary_heads)

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
        # 将 fuse_features 进行 bi_linear_upsample
        for i in self.HMSA_stage_choose:
            if i != 1:  # 上采样到 stage1 的大小
                fuse_features[i-1] = F.interpolate(fuse_features[i-1], (stage1_feature.size())[2:], mode='bilinear', align_corners=True)

        # -----------------------------------------------------------------
        # 利用 multi_loss_head 计算 multi_loss_scores
        multi_loss_scores = []
        for i in range(len(fuse_features)):
            stage_score = self.multi_loss_head[i](fuse_features[i])
            multi_loss_scores.append(stage_score)

        # -----------------------------------------------------------------
        # 利用 HMSA 的方式计算 final_score
        HMSA_stage_scores = []
        j = 0
        for i in self.HMSA_stage_choose:
            stage_score_map = multi_loss_scores[i-1]
            stage_attention_map = self.HMSA_attentions[j](fuse_features[i-1])
            HMSA_stage_score = stage_score_map*stage_attention_map           
 
            HMSA_stage_scores.append(HMSA_stage_score)
            j += 1
        final_score  = sum(HMSA_stage_scores)
       
        out = F.interpolate(final_score, x_size[2:], mode='bilinear', align_corners=True)

        # -----------------------------------------------------------------
        # 这个是判断是否用中间层 loss 进行辅助训练
        if self.training:
            if self.if_use_boundary_loss:
                segmask, boundarymask = y[0], y[1]  # 输入的各类 mask 解包
            else:
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

            # -------------------------------------------------------------------------
            # multi-stage loss 或 boundary loss
            if self.if_use_boundary_loss:
                # -------------------------------------------------------------------------
                # boundary loss
                for i in range(len(fuse_features)):   # 顺序：[stage1, stage2, stage3, stage4]
                    # -------------------------------------------
                    # 计算各 stage 的 boundary_score
                    boundary_score = self.boundary_heads[i](fuse_features[i])
                    boundary_score = F.interpolate(boundary_score, x_size[2:], mode='bilinear', align_corners=True)

                    # 各 stage 的 score head
                    multi_loss_score = F.interpolate(multi_loss_scores[i], x_size[2:], mode='bilinear', align_corners=True)

                    stage_loss = self.criterion((multi_loss_score, boundary_score),(segmask, boundarymask))  # 利用 GSCNN 计算 stage loss
                    loss += stage_loss
                return out.max(1)[1], loss
            else:
                # ----------------------------------------------------
                # multi-stage loss 的计算
                for i in range(len(multi_loss_scores)):   # 顺序：[stage1, stage2, stage3，stage4]
                        
                        stage_pred_out = F.interpolate(multi_loss_scores[i], x_size[2:], mode='bilinear', align_corners=True)
                        stage_loss = self.criterion(stage_pred_out, segmask) # 各 stage loss 的输出
                        loss += stage_loss

                return out.max(1)[1], loss                
        else:
            return out


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    