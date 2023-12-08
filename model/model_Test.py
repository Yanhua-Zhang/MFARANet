import torch
from torch import nn
import torch.nn.functional as F

from .backbone import resnet18_SFNet, resnet50_SFNet   # 加载预训练模型与参数
from .module import FAM_module, PPM

class Test_model(nn.Module):
    def __init__(self, use_dilation = True):
        super(Test_model, self).__init__()

        self.use_dilation = use_dilation

        # -------------------------------------------------------
        # Backbone
        resnet = resnet18_SFNet(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 # 分别对应 outchannel：[256,512,1024,2048]
        del resnet  # 这里删除变量名，释放内存

        # -------------------------------------------------------

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


        # -------------------------------------------------------
        # layer_test
        self.layer_test = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

        self.layer_test1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def forward(self, x, y=None):
        out = self.layer_test(x)
        out1 = self.layer_test1(x)
        
        return out, out1


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    