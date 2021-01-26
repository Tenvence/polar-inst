import copy

import torch.nn as nn


class PolarHead(nn.Module):
    def __init__(self, num_polars, num_channels, num_classes):
        super(PolarHead, self).__init__()

        conv_seq = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(),
        )
        self.cls_conv_seq = copy.deepcopy(conv_seq)
        self.reg_conv_seq = copy.deepcopy(conv_seq)

        self.cls_conv = nn.Conv2d(num_channels, num_classes, 3, padding=1)
        self.bbox_reg_conv = nn.Conv2d(num_channels, 4, 3, padding=1)
        self.centerness_conv = nn.Conv2d(num_channels, 1, 3, padding=1)

    def forward(self, x):
        cls_feature = self.cls_conv_seq(x)
        reg_feature = self.reg_conv_seq(x)

        cls = self.cls_conv(cls_feature)
        bbox = self.bbox_reg_conv(reg_feature)
        centerness = self.centerness_conv(reg_feature)

        return {'cls': cls, 'bbox': bbox, 'centerness': centerness}
