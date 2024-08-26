import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append("")
from util import BinLinear
from util import BinConv2d

cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Bin_VGG_train(nn.Module):
    def __init__(self, vgg_name):
        super(Bin_VGG_train, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
            ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool'+str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                layers['conv'+str(cnt)] = BinConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, istrain=False)
                cnt += 1
                layers['bn'+str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu'+str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool'+str(cnt)] = nn.AdaptiveAvgPool2d((1, 1))
        return nn.Sequential(layers)
class Bin_linear_test(nn.Module):
    def __init__(self):
        super(Bin_linear_test, self).__init__()
        self.classifier = BinLinear(10000, 64, istrain=False)

    def forward(self, x):
        out = self.classifier(x)
        return out   
class Bin_conv2d_test(nn.Module):
    def __init__(self):
        super(Bin_conv2d_test, self).__init__()
        self.features = BinConv2d(64, 64, kernel_size = 3, stride = 1, padding =1, istrain=False)

    def forward(self, x):
        out = self.features(x)
        return out      

class Conv2d_test(nn.Module):
    def __init__(self):
        super(Conv2d_test, self).__init__()
        self.features = nn.Conv2d(128, 1024, kernel_size = 3, stride = 1, padding =1)

    def forward(self, x):
        out = self.features(x)
        return out

class linear_test(nn.Module):
    def __init__(self):
        super(linear_test, self).__init__()
        self.classifier = nn.Linear(10000, 1)

    def forward(self, x):
        out = self.classifier(x)
        return out   
class Bin_VGG_test(nn.Module):
    def __init__(self, vgg_name):
        super(Bin_VGG_test, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
            ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool'+str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                layers['conv'+str(cnt)] = BinConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, istrain=False)
                cnt += 1
                layers['bn'+str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu'+str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool'+str(cnt)] = nn.AvgPool2d(kernel_size=1, stride=1)
        return nn.Sequential(layers)





class NIN_train(nn.Module):
    def __init__(self):
        super(NIN_train, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv2 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0, istrain=True)
        self.conv3 = BinConv2d(160, 96, kernel_size=1, stride=1, padding=0, istrain=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, istrain=True, drop=0.5)
        self.conv5 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=True)
        self.conv6 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=True)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv7 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, istrain=True, drop=0.5)
        self.conv8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=True)
        self.bn2 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        x = self.pool3(x)
        return x.view(x.size(0), 10)
class NIN_test(nn.Module):
    def __init__(self):
        super(NIN_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv2 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0, istrain=False)
        self.conv3 = BinConv2d(160, 96, kernel_size=1, stride=1, padding=0, istrain=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, istrain=False, drop=0.5)
        self.conv5 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=False)
        self.conv6 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=False)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv7 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, istrain=False, drop=0.5)
        self.conv8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0, istrain=False)
        self.bn2 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)
        x = self.conv7(x)
        x = F.relu(x, inplace=True)
        x = self.conv8(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.conv9(x)
        x = F.relu(x, inplace=True)
        x = self.pool3(x)
        return x.view(x.size(0), 10)

