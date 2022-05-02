import torch
import torch.nn as nn
from torch.nn import Conv2d, AvgPool2d, Unfold, Flatten
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torchvision import models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class Involution(nn.Module):

    '''
        Involution is used to compute the involution operation on the input
        activation map and generate the output activation map. The output
        activation map contains the same number of channels as the input
        activation map.

        channels: number of channels in the input activation map
        kernel_size: size of the kernel
        stride: stride along the axes
        reduction_ratio: if there are C channels, then the intermediate output contains C // r channels
        group_size: number of channels per group

        To compute the weights of the kernel, the input is passed through an average pooling layer
        if the stride is greater than one. It is then passed through reduce_channels_conv to produce
        an activation map of shape (batches, h, w, C // r). Next, the weights are passed through
        patch_conv to produce an output of shape (batches, h, w, k * k * G). The weights matrix is 
        now unfolded. A patch of similar image is obtained from the image and convolved with the
        unfolded weight matrix to produce the involution output.

    '''

    def __init__(self, channels, kernel_size, stride, reduction_ratio = 2, group_size = 4):

        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        self.group_size = group_size
        self.G = self.channels // self.group_size   # G represents the number of groups
        
        reduced_channels = self.channels // reduction_ratio
        out_channels = kernel_size * kernel_size * self.G

        self.reduce_channels_conv = Conv2d(channels, reduced_channels, 1)


        self.patch_conv = Conv2d(reduced_channels, out_channels, 1)
        
        
        self.pool_out = AvgPool2d(stride, stride)
        self.unfold = Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
    
    def forward(self, x):
        wt = x.clone().detach()
        if self.stride > 1:
            wt = self.pool_out(x)
        wt = self.reduce_channels_conv(wt)
        wt = self.patch_conv(wt)

        batches, channels, height, width = wt.shape

        wt = wt.view(batches, self.G, self.kernel_size * self.kernel_size, height, width)
        wt = wt.unsqueeze(2)
        
        patch = self.unfold(x)
        patch = patch.view(batches, self.G, self.group_size, self.kernel_size * self.kernel_size, height, width)

        output = (wt * patch).sum(dim = 3)
        output = output.view(batches, self.channels, height, width)
        return output


""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

##architecture_config = [
##    
##    (7, 64, 2, 3),
##    "M",
##    ("inv", 64, 3, 1),
##    (3, 192, 1, 1),
##    
##    "M",
##    (1, 128, 1, 0),
##    (3, 256, 1, 1),
##    (1, 256, 1, 0),
##    (3, 512, 1, 1),
##    
##    "M",
##    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
##    (1, 512, 1, 0),
##    (3, 1024, 1, 1),
##    "M",
##    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
##    (3, 1024, 1, 1),
##    (3, 1024, 2, 1),
##    (3, 1024, 1, 1),
##    (3, 1024, 1, 1),
##]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                if x[0] == 'inv':
                    print(x)
                    layers += [Involution(in_channels, x[1], x[2], x[3])]
                    #   in_channels will not change in involutions
                    continue

                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
