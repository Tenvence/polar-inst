import torch.nn as nn
import torch.nn.functional as func


class FeaturePyramidNet(nn.Module):
    def __init__(self, out_channels):
        super(FeaturePyramidNet, self).__init__()

        self.interval_conv_c3 = nn.Sequential(nn.Conv2d(512, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.interval_conv_c4 = nn.Sequential(nn.Conv2d(1024, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.interval_conv_c5 = nn.Sequential(nn.Conv2d(2048, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())

        self.fpn_conv_p3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fpn_conv_p4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fpn_conv_p5 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fpn_conv_p6 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.fpn_conv_p7 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, c3, c4, c5):
        interval_c3 = self.interval_conv_c3(c3)
        interval_c4 = self.interval_conv_c4(c4)
        interval_c5 = self.interval_conv_c5(c5)

        interval_c4 += func.interpolate(interval_c5, scale_factor=2, mode='nearest')
        interval_c3 += func.interpolate(interval_c4, scale_factor=2, mode='nearest')

        p3 = self.fpn_conv_p3(interval_c3)
        p4 = self.fpn_conv_p4(interval_c4)
        p5 = self.fpn_conv_p4(interval_c5)
        p6 = self.fpn_conv_p6(p5)
        p7 = self.fpn_conv_p7(p6)

        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'p7': p7}
