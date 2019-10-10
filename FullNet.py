"""
This script defines the structure of FullNet

Author: Hui Qu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvLayer, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=False, groups=groups))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


# --- different types of layers --- #
class BasicLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BasicLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation,
                              dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BottleneckLayer, self).__init__()

        inter_planes = growth_rate * 4
        self.conv1 = ConvLayer(in_channels, inter_planes, kernel_size=1, padding=0)
        self.conv2 = ConvLayer(inter_planes, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# --- dense block structure --- #
class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, layer_type, dilations):
        super(DenseBlock, self).__init__()
        for i in range(len(dilations)):
            layer = layer_type(in_channels+i*growth_rate, growth_rate, drop_rate, dilations[i])
            self.add_module('denselayer{:d}'.format(i+1), layer)


def choose_hybrid_dilations(n_layers, dilation_schedule, is_hybrid):
    import numpy as np
    # key: (dilation, n_layers)
    HD_dict = {(1, 4): [1, 1, 1, 1],
               (2, 4): [1, 2, 3, 2],
               (4, 4): [1, 2, 5, 9],
               (8, 4): [3, 7, 10, 13],
               (16, 4): [13, 15, 17, 19],
               (1, 6): [1, 1, 1, 1, 1, 1],
               (2, 6): [1, 2, 3, 1, 2, 3],
               (4, 6): [1, 2, 3, 5, 6, 7],
               (8, 6): [2, 5, 7, 9, 11, 14],
               (16, 6): [10, 13, 16, 17, 19, 21]}

    dilation_list = np.zeros((len(dilation_schedule), n_layers), dtype=np.int32)

    for i in range(len(dilation_schedule)):
        dilation = dilation_schedule[i]
        if is_hybrid:
            dilation_list[i] = HD_dict[(dilation, n_layers)]
        else:
            dilation_list[i] = [dilation for k in range(n_layers)]

    return dilation_list


class FullNet(nn.Module):
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), is_hybrid=True, layer_type='basic'):
        super(FullNet, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, is_hybrid)

        for i in range(n_blocks):  # no trans in last block
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block%d' % (i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans%d' % (i+1), trans)
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out


class FCN_pooling(nn.Module):
    """same structure with FullNet, except that there are pooling operations after block 1, 2, 3, 4
    and upsampling after block 5, 6
    """
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), hybrid=1, layer_type='basic'):
        super(FCN_pooling, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, hybrid)

        for i in range(7):
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block{:d}'.format(i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans{:d}'.format(i+1), trans)
            if i in range(0, 4):
                self.blocks.add_module('pool{:d}'.format(i+1), nn.MaxPool2d(kernel_size=2, stride=2))
            elif i in range(4, 6):
                self.blocks.add_module('upsample{:d}'.format(i + 1), nn.UpsamplingBilinear2d(scale_factor=4))
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out
