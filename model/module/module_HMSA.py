import torch
import torch.nn as nn
# from torch.nn import Module, Parameter
import torch.nn.functional as F

__all__ = ['HMSA_attention_Head']

class HMSA_attention_Head(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, norm_layer):
        super(HMSA_attention_Head, self).__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.attn(x)