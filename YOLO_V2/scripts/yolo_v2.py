import torch
import torch.nn as nn

from utils import make_conv_layers

# tuple -> (out_channels, kernel_size)
# M -> MaxPool
DARKNET_BACKBONE = {
    "stage1_conv": [
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
    ],
    "stage2_conv": ["M", (1024, 3), (512, 1), (1024, 3), (512, 3), (1024, 3)],
    "fcn_layer_in_channel": 3072,
    "fcn_layers": [(1024, 3), (1024, 3), (1024, 3)],
}



class YOLOv2(nn.Module):
    def __init__(
        self, backbone_config=DARKNET_BACKBONE, no_of_classes=C, no_of_anchor_box=N
    ):
        super().__init__()
        self.in_channels = 3
        self.arch_config = backbone_config
        self.no_of_anchor_box = no_of_anchor_box
        self.no_of_classes = no_of_classes
        self.output_layer_in_channels = self.arch_config["fcn_layers"][-1][0]

        # no of anchor boxes * (4 bb + 4 class prob + object confidence score)*13*13 for dataset with 4 classes
        self.output_channels = self.no_of_anchor_box * (self.no_of_classes + 1 + 4)

        # Conv Layers
        self.stage1_conv_layers = make_conv_layers(self.arch_config["stage1_conv"])
        self.stage2_conv_layers = make_conv_layers(
            self.arch_config["stage2_conv"],
            in_channels=self.arch_config["stage1_conv"][-1][0],
        )
        self.fcn_layers = make_conv_layers(
            self.arch_config["fcn_layers"],
            in_channels=self.arch_config["fcn_layer_in_channel"],
        )

        self.ouput_layer = nn.Conv2d(
            in_channels=self.output_layer_in_channels,
            out_channels=self.output_channels,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        x1 = self.stage1_conv_layers(x)
        x2 = self.stage2_conv_layers(x1)

        # Skip connection from stage 1: slice into 4 parts and concatenate
        _, _, height, width = x1.size()

        part1 = x1[:, :, : height // 2, : width // 2]
        part2 = x1[:, :, : height // 2, width // 2 :]
        part3 = x1[:, :, height // 2 :, : width // 2]
        part4 = x1[:, :, height // 2 :, width // 2 :]
        residual = torch.cat((part1, part2, part3, part4), dim=1)

        # Concatenate residual with x2
        x_concat = torch.cat((x2, residual), dim=1)

        # Pass through FCN layers
        x3 = self.fcn_layers(x_concat)

        # Pass through classifier
        
        out = self.ouput_layer(x3)
        
        #reshaping the ouput to have a ouput  for every anchor box at the end so it will be easy to acess) #B,S,S,5,9
        new_out = out.permute(0,2,3,1).contiguous()
        
        return new_out.view(new_out.size(0),new_out.size(1),new_out.size(2), self.no_of_anchor_box, 5+self.no_of_classes)
