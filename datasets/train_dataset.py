import os

import albumentations as alb
import torch
import torch.nn.functional as func
import torchvision.datasets as cv_datasets
import torchvision.ops as cv_ops
from albumentations.pytorch.transforms import ToTensorV2

import datasets.tools as tools
import utils.bbox_ops as bbox_ops


class TrainDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size):
        super(TrainDataset, self).__init__(root=os.path.join(root, 'train'), annFile=os.path.join(root, 'train.json'))

        self.h, self.w = input_size

        self.transform = alb.Compose([
            alb.RandomSizedBBoxSafeCrop(width=self.w, height=self.h),
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0., p=0.8),
            alb.Normalize(),
            ToTensorV2()
        ], bbox_params=alb.BboxParams(format='coco', label_fields=['class_labels']))

    def __getitem__(self, index):
        img, target = tools.load_img_target(self, index)

        cls_labels = [obj['category_id'] for obj in target]
        bbox_labels = [obj['bbox'] for obj in target]

        transformed = self.transform(image=img, bboxes=bbox_labels, class_labels=cls_labels)
        img = transformed['image']
        cls_labels = torch.as_tensor(transformed['class_labels'])
        bbox_labels = cv_ops.box_convert(torch.as_tensor(transformed['bboxes']), in_fmt='xywh', out_fmt='xyxy')

        all_level_points, class_targets, distance_targets = self._encode_targets(cls_labels, bbox_labels)
        centerness_targets = self._encode_centerness_targets(distance_targets)

        return img, {
            'points': all_level_points,
            'class_targets': class_targets,
            'distance_targets': distance_targets,
            'centerness_targets': centerness_targets
        }

    @staticmethod
    def _encode_centerness_targets(distance_targets):
        left_right = distance_targets[:, [0, 2]]
        top_bottom = distance_targets[:, [1, 3]]
        centerness_targets = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    def _encode_targets(self, cls_labels, bbox_labels):
        all_level_points, regress_ranges = tools.encode_all_level_points(self.h, self.w)

        num_points = all_level_points.size(0)
        num_gts = cls_labels.size(0)

        regress_ranges = regress_ranges[:, None, :].repeat(1, num_gts, 1)  # [num_points, num_gts, 2]
        bbox_areas = cv_ops.box_area(bbox_labels)[None].repeat(num_points, 1)  # [num_points, num_gts]

        expanded_points = all_level_points[:, None, :].repeat(1, num_gts, 1)
        expanded_bboxes = bbox_labels[None, :, :].repeat(num_points, 1, 1)
        distance_targets = bbox_ops.convert_bbox_to_distance(expanded_points, expanded_bboxes)  # [num_points, num_gts, 4]

        # Condition 1: inside a gt bbox
        inside_gt_bbox_mask = distance_targets.min(dim=-1)[0] > 0  # [num_points, num_gts]

        # Condition 2: limit the regression range for each location
        max_regress_distance = distance_targets.max(dim=-1)[0]  # [num_points, num_gts]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])  # [num_points, num_gts]

        # If there are still more than one instances for a location, we choose the one with minimal area
        bbox_areas[inside_gt_bbox_mask == 0] = tools.INF
        bbox_areas[inside_regress_range == 0] = tools.INF
        min_area, min_area_idx = bbox_areas.min(dim=1)  # [num_points], Assign a gt to each location

        class_targets = cls_labels[min_area_idx]
        class_targets[min_area == tools.INF] = 0

        distance_targets = distance_targets[range(num_points), min_area_idx, :]

        return all_level_points, class_targets, distance_targets
