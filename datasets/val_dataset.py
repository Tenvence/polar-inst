import os

import albumentations as alb
import torchvision.datasets as cv_datasets
from albumentations.pytorch.transforms import ToTensorV2

import datasets.tools as tools


class ValDataset(cv_datasets.CocoDetection):
    def __init__(self, root, input_size):
        super(ValDataset, self).__init__(root=os.path.join(root, 'val'), annFile=os.path.join(root, 'val.json'))
        self.h, self.w = input_size
        self.transform = alb.Compose([
            alb.Resize(width=self.w, height=self.h),
            alb.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        img, target = tools.load_img_target(self, index)

        img = self.transform(image=img)['image']
        img_id = self.ids[index]

        all_level_points, _ = tools.encode_all_level_points(self.h, self.w)

        return img, {'img_id': img_id, 'points': all_level_points}
