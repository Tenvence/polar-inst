import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.ops as cv_ops

import datasets.tools as tools
import utils.bbox_ops as bbox_ops


class FcosLoss(nn.Module):
    def __init__(self, focal_alpha, focal_gamma):
        super(FcosLoss, self).__init__()

        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, pred, target, points):
        class_pred, distance_pred, centerness_pred = pred['class'], pred['distance'], pred['centerness']
        class_targets, distance_targets, centerness_targets = target['class'], target['distance'], target['centerness']

        positive_idx = torch.nonzero(class_targets.reshape(-1)).reshape(-1)
        pos_distance_pred = distance_pred.reshape(-1, 4)[positive_idx]  # [num_positives, 4]
        pos_distance_targets = distance_targets.reshape(-1, 4)[positive_idx]  # [num_positives, 4]
        pos_centerness_pred = centerness_pred.reshape(-1)[positive_idx]  # [num_positives]
        pos_centerness_targets = centerness_targets.reshape(-1)[positive_idx]  # [num_positives]

        pos_points = points.reshape(-1, 2)[positive_idx]
        pos_decoded_bbox_pred = bbox_ops.convert_distance_to_bbox(pos_points, pos_distance_pred)
        pos_decoded_bbox_targets = bbox_ops.convert_distance_to_bbox(pos_points, pos_distance_targets)

        class_targets = func.one_hot(class_targets, num_classes=len(tools.VOC_CLASSES) + 1).float()
        bg_targets = class_targets[..., 0]
        fg_class_targets = class_targets[..., 1:]
        loss_cls = cv_ops.sigmoid_focal_loss(class_pred, fg_class_targets, self.alpha, self.gamma, reduction='sum') / (1. - bg_targets).sum()

        iou_loss = -cv_ops.box_iou(pos_decoded_bbox_pred, pos_decoded_bbox_targets).diagonal().clamp(min=1e-6).log()
        # iou_loss = 1 - cv_ops.generalized_box_iou(pos_decoded_bbox_pred, pos_decoded_bbox_targets).diagonal()
        loss_bbox = (pos_centerness_targets * iou_loss).sum() / pos_centerness_targets.sum()

        loss_centerness = func.binary_cross_entropy_with_logits(pos_centerness_pred, pos_centerness_targets)

        return loss_cls, loss_bbox, loss_centerness
