import os
import json
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.tensorboard as tensorboard
import tqdm

import utils.eval


class DistributedLogger:
    def __init__(self, name, master_rank, use_tensorboard=True):
        self.name = name

        self.master_rank = master_rank
        self.is_master_rank = dist.get_rank() == self.master_rank

        self.use_tensorboard = self.is_master_rank and use_tensorboard
        self.tensorboard_logger = tensorboard.SummaryWriter(comment=name) if self.use_tensorboard else None

        self.model_path, self.heat_map_path, self.val_path = self._build_file_structure()

    def _build_file_structure(self, base_dir='./output'):
        if not os.path.exists(base_dir) and self.is_master_rank:
            os.mkdir(base_dir)

        output_path = os.path.join(base_dir, self.name)
        if not os.path.exists(output_path) and self.is_master_rank:
            os.mkdir(output_path)

        model_path = os.path.join(output_path, 'model')
        if not os.path.exists(model_path) and self.is_master_rank:
            os.mkdir(model_path)

        heat_map_path = os.path.join(output_path, 'heat_map')
        if not os.path.exists(heat_map_path) and self.is_master_rank:
            os.mkdir(heat_map_path)

        val_path = os.path.join(output_path, 'val')
        if not os.path.exists(val_path) and self.is_master_rank:
            os.mkdir(val_path)

        return model_path, heat_map_path, val_path

    def save_model(self, model):
        if not self.is_master_rank:
            return

        torch.save(model.module, os.path.join(self.model_path, 'model.pkl'))
        torch.save(model.module.state_dict(), os.path.join(self.model_path, 'param.pth'))

    def reduce_tensor(self, tensor):
        dist.reduce(tensor, dst=self.master_rank, op=dist.reduce_op.SUM)
        return tensor.item() / dist.get_world_size()

    def reduce_epoch_loss(self, loss_list):
        avg_loss_one_device = torch.stack(loss_list, dim=-1).mean()
        return self.reduce_tensor(avg_loss_one_device)

    def update_tensorboard(self, super_tag, tag_scaler_dict, idx):
        if not self.use_tensorboard:
            return

        for tag, scaler in tag_scaler_dict.items():
            self.tensorboard_logger.add_scalar(super_tag + '/' + tag, scaler, idx)

    def init_processor(self, data_loader):
        return tqdm.tqdm(data_loader) if self.is_master_rank else data_loader

    def update_processor(self, processor, description_str):
        if not self.is_master_rank:
            return

        processor.set_description(description_str)

    def save_pred_instances_local_rank(self, pred_instances):
        local_rank = dist.get_rank()
        np.save(os.path.join(self.val_path, f'tmp_pred_instances_rank_{local_rank}.npy'), pred_instances)

    def save_val_file(self):
        if not self.is_master_rank:
            return

        time.sleep(5.)  # wait all threads to finish
        pred_instances = []

        for local_rank in range(dist.get_world_size()):
            tmp_pre_instances_file = os.path.join(self.val_path, f'tmp_pred_instances_rank_{local_rank}.npy')
            pred_instances.extend(np.load(tmp_pre_instances_file, allow_pickle=True))
            os.remove(tmp_pre_instances_file)

        json.dump(pred_instances, open(os.path.join(self.val_path, 'val.json'), 'w'))

    def update_tensorboard_val_results(self, coco_gt, epoch_idx):
        if not self.is_master_rank:
            return

        if len(json.load(open(os.path.join(self.val_path, 'val.json'), 'r'))) == 0:
            print('no prediction!')
            ap, ap50, ap75, ap_s, ap_m, ap_l, ar1, ar10, ar100, ar_s, ar_m, ar_l = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            ap, ap50, ap75, ap_s, ap_m, ap_l, ar1, ar10, ar100, ar_s, ar_m, ar_l = utils.eval.bbox_evaluate(coco_gt, os.path.join(self.val_path, 'val.json'))

        self.update_tensorboard(super_tag='segm-val', tag_scaler_dict={
            'AP': ap * 100, 'AP@50': ap50 * 100, 'AP@75': ap75 * 100,
            'AP@S': ap_s * 100, 'AP@M': ap_m * 100, 'AP@L': ap_l * 100,
            'AR@1': ar1 * 100, 'AR@10': ar10 * 100, 'AR@100': ar100 * 100,
            'AR@S': ar_s * 100, 'AR@M': ar_m * 100, 'AR@L': ar_l * 100
        }, idx=epoch_idx)
