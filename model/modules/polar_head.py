import copy

import numpy as np
import torch.nn as nn


class PolarHead(nn.Module):
    def __init__(self, num_polars, num_channels, num_classes):
        super(PolarHead, self).__init__()

        conv_seq = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.GroupNorm(32, num_channels), nn.ReLU(inplace=True),
        )

        self.cls_conv_seq = copy.deepcopy(conv_seq)
        self.reg_conv_seq = copy.deepcopy(conv_seq)

        self.cls_conv = nn.Conv2d(num_channels, num_classes, 3, padding=1)
        self.distance_reg_conv = nn.Conv2d(num_channels, 4, 3, padding=1)
        self.centerness_conv = nn.Conv2d(num_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.cls_conv_seq:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        for m in self.reg_conv_seq:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

        cls_bias_prior = 0.01
        nn.init.normal_(self.cls_conv.weight, std=0.01)
        nn.init.constant_(self.cls_conv.bias, val=float(-np.log((1 - cls_bias_prior) / cls_bias_prior)))

        nn.init.normal_(self.distance_reg_conv.weight, std=0.01)
        nn.init.constant_(self.distance_reg_conv.bias, val=0)

        nn.init.normal_(self.centerness_conv.weight, std=0.01)
        nn.init.constant_(self.centerness_conv.bias, val=0)

    def forward(self, x):
        cls_feature = self.cls_conv_seq(x)
        reg_feature = self.reg_conv_seq(x)

        cls = self.cls_conv(cls_feature)
        distances = self.distance_reg_conv(reg_feature)
        centerness = self.centerness_conv(reg_feature)

        return {'cls': cls, 'distance': distances, 'centerness': centerness}
