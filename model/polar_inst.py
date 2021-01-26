import torch
import torch.nn as nn

from .stage_backbone import StageBackbone
from .feature_pyramid_net import FeaturePyramidNet
from .polar_head import PolarHead


class PolarInst(nn.Module):
    def __init__(self, num_polars, num_channels, num_classes):
        super(PolarInst, self).__init__()

        self.backbone = StageBackbone()
        self.fpn = FeaturePyramidNet(num_channels)
        self.polar_head = PolarHead(num_polars, num_channels, num_classes)

        self.bbox_scales = [nn.Parameter(torch.tensor(1., dtype=torch.float)) for _ in range(5)]

    def forward(self, x):
        backbone_outs = self.backbone(x)
        fpn_outs = self.fpn(backbone_outs['c3'], backbone_outs['c4'], backbone_outs['c5'])

        out = {}
        for idx, (bbox_scale, fpn_out) in enumerate(zip(self.bbox_scales, fpn_outs.values())):
            head_out = self.polar_head(fpn_out)

            head_out['bbox'] *= bbox_scale
            head_out['bbox'] = head_out['bbox'].float().exp()

            out[f'p{idx + 3}'] = head_out

        return out
