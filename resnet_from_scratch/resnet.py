import torch
import torch.nn as nn

from resnet_from_scratch.block import Bottleneck


class ResNet(nn.Module):

    def __init__(self, resnet_variant, in_channels, num_classes):
        """
        Creates the ResNet architecture based on the provided variant. 18/34/50/101 etc.
        Based on the input parameters, define the channels list, repetition list along with expansion factor(4) and stride(3/1)
        using _make_blocks method, create a sequence of multiple Bottlenecks
        Average Pool at the end before the FC layer

        Args:
            resnet_variant (list): eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_channels (int): image channels (3)
            num_classes (int): output classes

        Attributes:
            Layer consisting of conv->batchnorm->relu
        """
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repetition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repetition_list[0], self.expansion, self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1], self.repetition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2], self.repetition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3], self.repetition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)
        
    def _make_blocks(self, in_channels, intermediate_channels, num_repeats, expansion, is_Bottleneck, stride):
        """
        Args:
            in_channels: channels of the Bottleneck input
            intermediate_channels: channels of the 3x3 in the Bottleneck
            num_repeats: Bottlenecks in the block
            expansion: factor by which intermediate_channels are multiplied to create the output channels
            is_Bottleneck: status if Bottleneck in required
            stride: stride to be used in the first Bottleneck conv 3x3
        
            Attributes:
                Sequence of Bottleneck layers
        """
        layers = []

        layers.append(
            Bottleneck(
                in_channels=in_channels,
                intermediate_channels=intermediate_channels,
                expansion=expansion,
                is_Bottleneck=is_Bottleneck,
                stride=stride
            )
        )
        for num in range(1, num_repeats):
            layers.append(
                Bottleneck(
                    in_channels=intermediate_channels * expansion,
                    intermediate_channels=intermediate_channels,
                    expansion=expansion,
                    is_Bottleneck=is_Bottleneck,
                    stride=1
                )
            )
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

    