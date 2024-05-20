import torch.nn as nn
from scripts.utils import make_conv_layers

# tuple = no of fileter, kernel
DARKNET = [
    (32, 3),
    "M",
    (64, 3),
    "M",
    (128, 3),
    (64, 1),
    (128, 1),
    "M",
    (256, 3),
    (128, 1),
    (256, 3),
    "M",
    (512, 3),
    (256, 1),
    (512, 3),
    (256, 1),
    (512, 3),
    "M",
    (1024, 3),
    (512, 1),
    (1024, 3),
    (512, 3),
    (1024, 3),
]


class DarkNet(nn.Module):
    def __init__(self, arch_config=DARKNET, no_of_classes=1000):
        super().__init__()
        self.arch_config = arch_config
        self.in_channels = 3
        self.no_of_classes = no_of_classes
        self.conv_layers = make_conv_layers(self.arch_config)

        self.fc_in_channels = arch_config[-1][0]

        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.fc_in_channels,
                no_of_classes,
                kernel_size=1,
                padding="same",
                bias=False,
            ),  # not using bias as batchnorm
            nn.BatchNorm2d(no_of_classes),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
