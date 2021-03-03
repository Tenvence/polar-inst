import argparse
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist
import torch.nn.parallel as parallel
import torch.optim as optim
import torch.utils.data as data

from datasets import VOC_CLASSES, TrainDataset, ValDataset
from engine import train_one_epoch, val_one_epoch
from model import PolarInst, FcosLoss
from utils.distributed_logger import DistributedLogger
from utils.lr_lambda import get_warm_up_multi_step_lr_lambda

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', default=True, type=bool)

    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--random_seed', default=970423, type=int)

    # parser.add_argument('--input_size_h', default=768, type=int)
    # parser.add_argument('--input_size_w', default=1280, type=int)
    parser.add_argument('--input_size_h', default=512, type=int)
    parser.add_argument('--input_size_w', default=512, type=int)
    parser.add_argument('--num_polars', default=36, type=int)
    parser.add_argument('--num_channels', default=256, type=int)

    parser.add_argument('--focal_alpha', default=.25, type=float)
    parser.add_argument('--focal_gamma', default=2., type=float)

    parser.add_argument('--epochs', default=24, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--bias_lr_mul', default=2., type=float)
    parser.add_argument('--bias_weight_decay_mul', default=0., type=float)
    parser.add_argument('--warm_up_epoch', default=1, type=int)
    parser.add_argument('--warm_up_ratio', default=1 / 3, type=float)
    parser.add_argument('--milestones', default=[14, 20], type=list)
    parser.add_argument('--step_gamma', default=0.1, type=float)

    parser.add_argument('--nms_pre', default=1000, type=int)
    parser.add_argument('--nms_cls_score_thr', default=0.05, type=int)
    parser.add_argument('--nms_iou_thr', default=0.5, type=int)

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--master_rank', default=0, type=int)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def __main__():
    args = get_args_parser()
    dist.init_process_group(backend='nccl')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.random_seed + dist.get_rank())
    torch.cuda.set_device(torch.device('cuda:{}'.format(dist.get_rank())))
    dist_logger = DistributedLogger(args.name, args.master_rank, use_tensorboard=False)

    train_dataset = TrainDataset(args.dataset_root, (args.input_size_h, args.input_size_w))
    train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)

    val_dataset = ValDataset(args.dataset_root, (args.input_size_h, args.input_size_w))
    val_sampler = data.distributed.DistributedSampler(val_dataset)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

    model = PolarInst(args.num_polars, args.num_channels, len(VOC_CLASSES)).cuda()
    # model.load_state_dict(torch.load(f'./output/{args.name}/model/param.pth'))
    model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    criterion = FcosLoss(args.focal_alpha, args.focal_gamma)

    optim_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if not n.endswith('bias') and p.requires_grad]},
        {
            'params': [p for n, p in model.module.named_parameters() if n.endswith('bias') and p.requires_grad],
            'lr': args.lr * args.bias_lr_mul,
            'weight_decay': args.weight_decay * args.bias_weight_decay_mul
        }
    ]
    optimizer = optim.SGD(optim_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = get_warm_up_multi_step_lr_lambda(len(train_dataloader), args.warm_up_epoch, args.warm_up_ratio, args.milestones, args.step_gamma)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    nms_cfg = {'nms_pre': args.nms_pre, 'cls_score_thr': args.nms_cls_score_thr, 'iou_thr': args.nms_iou_thr}

    for epoch_idx in range(args.epochs):
        train_sampler.set_epoch(epoch_idx)
        val_sampler.set_epoch(epoch_idx)

        train_one_epoch(model, optimizer, criterion, lr_scheduler, train_dataloader, dist_logger, epoch_idx)
        val_one_epoch(model, val_dataloader, val_dataset.coco, dist_logger, epoch_idx, nms_cfg)


if __name__ == '__main__':
    __main__()
