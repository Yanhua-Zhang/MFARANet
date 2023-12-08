import torch
import torch.nn as nn
# from torch.nn import Module, Parameter
import torch.nn.functional as F
from ..function.function_EncNet import scaled_l2, aggregate, Mean

__all__ = ['EncHead']

class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K    # D 是 channels，K 是 codewords
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)  # 这指的应该是 K 个视觉中心
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)     # K 个系数
        self.reset_params()

    def reset_params(self):    # 进行参数值初始化
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)  # .uniform_ ：从 -std1,std1 均匀分布中抽取样本进行填充
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            # Tensor不连续的，会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作
            X = X.view(B, D, -1).transpose(1, 2).contiguous()  # 维度调整，方便与 K 个 1xD 的视觉中心向量作差
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        # 输入 X: b, C, h, w。视觉中心：KxC。系数：Kx1。
        # 输出： BxNxK
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)   
        # aggregate
        # 输出：BxKxC
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),   # encoding 操作
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))       # 按输入 tensor 的列计算均值
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)   # 利用全连接层，判断 n 个类别的出现概率

    def forward(self, x):
        en = self.encoding(x)                 # 经过 encoding 操作：BxKxC ？？？BxC???
        b, c, _, _ = x.size()
        gamma = self.fc(en)                   # 全连接层+sigmod: 这里输入、输出都应该是 BxC
        y = gamma.view(b, c, 1, 1)            # 
        # outputs = [F.relu_(x + x * y)]        # relu 激活。 x*y 是对通道作一个类别筛选
        output1 = F.relu_(x + x * y)
        if self.se_loss:
            # outputs.append(self.selayer(en))  # 计算 se_loss 时，用 en 作为输入
            output2 = self.selayer(en)
        # return tuple(outputs)
        return output1, output2


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, lateral=True,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))    # 用于进行 channel 的调整
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:                      # 是否进行 skip connect 的操作
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))    # 这里各 stage 的 size 是一样的：1/8

        # outs = list(self.encmodule(feat))     # 将 feature map 输入 EncModule 
        out1, out2 = self.encmodule(feat)
        out1 = self.conv6(out1)         # 计算 score map
        return out1, out2  # outs[0]：score map、outs[1]：se map
