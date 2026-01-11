from typing import List, Tuple
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


class Transition(nn.Module):  # BN-ConV-Pooling  在每一个block之间  通道数减半
    def __init__(self, in_channel, p=0.5):
        """
        :param in_channel:
        :param p: block_out_channel * p = transition_out_channel (0 < p < 1)
        """
        super(Transition, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_feature: int,  # the number of input channel
            growth_rate: int,  # each function H` produces k feature maps, H' just is DenseLayer
            drop_rate: float,  # dropout rate
            bn_size: int  # bottleneck to reduce the number of input feature-maps
    ):
        super(DenseLayer, self).__init__()
        # Bottleneck Layer
        self.norm1 = nn.BatchNorm2d(num_input_feature)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_feature, growth_rate * bn_size, kernel_size=(1, 1), stride=(1, 1))
        """
        paper:
            a 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution
            to reduce the number of input feature-maps, and thus to improve computational efficiency
        """
        # Composite Function
        self.norm2 = nn.BatchNorm2d(growth_rate * bn_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate * bn_size, growth_rate, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        """ 
        paper:
            each function H` produces k feature maps, growth_rate just is k
        """

        self.drop_rate = float(drop_rate)

    def forward(self, x: List[Tensor]):  # x.size=[b, c, h, w] 最终输出 ： 尺寸不变，通道数变为growth_rate
        features = torch.cat(x, dim=1)  # xl = H([x0, x1, x2....xl-1])
        bottleneck_out = self.conv1(self.relu1(self.norm1(features)))
        h_out = self.conv2(self.relu2(self.norm2(bottleneck_out)))
        layer_out = F.dropout(h_out, p=self.drop_rate)
        return layer_out


class DenseBlock(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_input_feature: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            # 每一层的输入通道大小都是前i层的growth_rate*i个加上初始输入通道，i从0开始
            layer = DenseLayer(
                num_input_feature=num_input_feature + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                bn_size=bn_size
            )  # 每一个层都会获得growth_rate个特征
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):  # [b, c, h, w]  每一个denselayer都不会改变尺寸，故整个denseblock也不会改变尺寸
        features = [init_features]  # features 是前面所有子层的feature
        for name, layer in self.items():
            """
            paper:
                 we introduce direct connections from any layer to all subsequent layers
                 x` = H` ([x0, x1, . . . , x` −1]),
                 the `th layer receives the feature-maps of all preceding layers,x0, . . . , x` −1
            """
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(tensors=features, dim=1)  # 一个block结束，将所得的features在C这个维度拼接起来


class DenseNet121(nn.Module):
    def __init__(
            self,
            growth_rate: int = 32,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_class: int = 1000,
            each_block_layer: Tuple[int, int, int, int] = (6, 12, 24, 16)
    ):
        super(DenseNet121, self).__init__()
        """
        paper:
            Before entering the first dense block, a convolution with 16 
            (or twice the growth rate for DenseNet-BC) output channels is performed on the input images
            Refer to Table 1
        """
        # First Convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, growth_rate * 2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))),
                    ("norm0", nn.BatchNorm2d(growth_rate * 2)),
                    ("relu0", nn.ReLU(True)),
                    ("pool0", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)))
                ]
            )
        )  # 尺寸减半
        # DenseBlocks
        num_features = growth_rate * 2
        for i, num_layer in enumerate(each_block_layer):
            block = DenseBlock(
                num_layers=num_layer,
                num_input_feature=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            num_features = num_features + growth_rate * (num_layer - 1)
            """ the lth layer has k0 +k × (l − 1) input feature-maps
            """
            self.features.add_module("denseblock%d" % (i + 1), block)
            if i != len(each_block_layer) - 1:  # 若不是最后一个block，则后面跟Transition
                transition = Transition(num_features)
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = num_features // 2
        # Final layer
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_class)

    def forward(self, x):  # [B, C, H, W]
        features = self.features(x)  # [B, num_feature, H/2, W/2]
        out = F.relu(features, inplace=True)
        # 自适应池化，给出想要的尺寸，会自动计算池化核以及步长
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))  # [B, num_feature, 1, 1]
        out = torch.flatten(out, 1)  # [B, num_feature*1*1]
        out = self.classifier(out)  # [B, class_num]
        return out


def main():
    densenet = DenseNet121()
    print(densenet)


if __name__ == '__main__':
    main()