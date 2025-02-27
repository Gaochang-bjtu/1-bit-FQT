import torch.nn as nn
from util import BinConv2d

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
            ) -> nn.Module:
    """3x3 convolution with padding"""
    return BinConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False)

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1
            ) -> nn.Module:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
