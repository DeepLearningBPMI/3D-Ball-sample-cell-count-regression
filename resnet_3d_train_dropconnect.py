#pretrained model MadicalNet https://github.com/Tencent/MedicalNet
#replace last fully connected layer with global average pooling, drop out, 5 unit fully connected layer, softmax activation


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = torch.nn.functional.avg_pool3d(x, kernel_size =1, stride=stride)
    
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class Dropconnect(nn.Module):
    def __init__(self, input_dim, output_dim, drop_prob=0.3):
        super(Dropconnect,self).__init__()
        self.drop_prob = drop_prob
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
    def forward(self, x):
        if self.training:
            #Generate a mask of the same shape as the weights
            mask = torch.rand(self.weight.size())>self.drop_prob
            #Apply Dropconnect: multiply weight by mask
            drop_weight = self.weight * mask.float().to(self.weight.device)
        else:
            #Do not apply Dropconnect when testing, but adjust weights to reflect dropout rates
            drop_weight=self.weight*(1-self.drop_prob)
        return F.linear(x, drop_weight, self.bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, dilation=dilation,  bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,stride=1, padding=1, dilation=dilation,  bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 D, 
                 W, 
                 H,
                 drop_prob,
                 channels=3, 
                 num_output_classes = 4,
                 shortcut_type='A',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            channels, #number of input channels
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False 
            )
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, dilation=1)
        self.global_avg_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))   
        # self.fc1 = nn.Linear(512*block.expansion, 32)
        # self.dropout = nn.Dropout(dropout) 
        self.dropconnet_layer = Dropconnect(512*block.expansion, 32, drop_prob)
        self.output = nn.Linear(32, num_output_classes)
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), 
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        print("Shape before Dropconnect:", x.shape)
        # x = self.fc1(x)
        # x = self.dropout(x)
        x = self.dropconnet_layer (x)
        # x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x) 

        return x

def resnet10(D, W, H, dropout,channels = 3, **kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], D, W, H, dropout, channels = 3, **kwargs)
    return model

def resnet18(D, W, H,  dropout,channels = 3,**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],D, W, H,   dropout,channels = 3, **kwargs)
    return model


def resnet34(D, W, H,  dropout, channels = 3,**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], D, W, H,  dropout,channels = 3,**kwargs)
    return model


def resnet50(D, W, H, dropout,channels = 3,**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], D, W, H, dropout,channels = 3,**kwargs)
    return model


def resnet101(D, W, H,  dropout,channels = 3,**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],D, W, H, dropout,channels = 3, **kwargs)
    return model


def resnet152(D, W, H,  dropout,channels = 3,**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],D, W, H,   dropout,channels = 3, **kwargs)
    return model


def resnet200(D, W, H,  dropout,channels = 3,**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], D, W, H,  dropout,channels = 3,**kwargs)
    return model