import os

import cv2
import torch
import torchvision.datasets as cv_datasets

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

INF = 1e8
STRIDES = [8, 16, 32, 64, 128]
REGRESS_RANGES = [(-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)]


def load_img_target(dataset: cv_datasets.CocoDetection, index: int):
    coco = dataset.coco
    img_id = dataset.ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)

    target = coco.loadAnns(ann_ids)
    img = cv2.imread(os.path.join(dataset.root, coco.loadImgs(img_id)[0]['file_name']))

    return img, target


def get_level_points(h, w, stride):
    fm_h, fm_w = h // stride, w // stride
    x_range = torch.arange(0, fm_w * stride, stride)
    y_range = torch.arange(0, fm_h * stride, stride)

    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1) + stride // 2

    return points


def encode_all_level_points(h, w):
    all_level_points, regress_ranges = [], []
    for stride, regress_range in zip(STRIDES, REGRESS_RANGES):
        level_points = get_level_points(h, w, stride)
        all_level_points.append(level_points)

        regress_range = torch.tensor(regress_range)[None].repeat(level_points.size(0), 1)
        regress_ranges.append(regress_range)

    featmap_locations = torch.cat(all_level_points, dim=0)  # [num_points, 2]
    regress_ranges = torch.cat(regress_ranges, dim=0)  # [num_points, 2]

    return featmap_locations, regress_ranges
