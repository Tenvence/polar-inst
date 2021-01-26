# import argparse
# import random
# import warnings
#
# import numpy as np
import torch

# import torch.backends.cudnn
# import torch.distributed as dist
# import torch.nn.parallel as parallel
# import torch.optim as optim
# import torch.utils.data as data
#
# import engine
# from datasets import AugVocTrainDataset, AugVocValDataset, CLASSES
# from utils.distributed_logger import DistributedLogger
#
# warnings.filterwarnings('ignore')
#
#
# def get_args_parser():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--is_train', default=True, type=bool)
#
#     parser.add_argument('--dataset_root', type=str)
#     parser.add_argument('--name', type=str)
#     parser.add_argument('--random_seed', default=970423, type=int)
#
#     parser.add_argument('--input_size', default=448, type=int)
#     parser.add_argument('--num_instances', default=40, type=int)
#     parser.add_argument('--d_model', default=512, type=int)
#
#     parser.add_argument('--class_weight', default=1., type=float)
#     parser.add_argument('--giou_weight', default=2., type=float)
#     parser.add_argument('--l1_weight', default=5., type=float)
#     parser.add_argument('--no_instance_coef', default=0.1, type=float)
#
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--step_size', default=200, type=int)
#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--num_workers', default=4, type=int)
#
#     parser.add_argument('--lr', default=2e-4, type=float)
#     parser.add_argument('--weight_decay', default=1e-4, type=float)
#
#     parser.add_argument('--local_rank', default=0, type=int)
#     parser.add_argument('--master_rank', default=0, type=int)
#
#     return parser.parse_args()
#
#
# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)
#
#
# def __main__():
#     args = get_args_parser()
#     dist.init_process_group(backend='nccl')
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     set_random_seed(args.random_seed + dist.get_rank())
#     torch.cuda.set_device(torch.device('cuda:{}'.format(dist.get_rank())))
#     dist_logger = DistributedLogger(args.name, args.master_rank, use_tensorboard=True)
#
#     model = InstanceTransformer(args.num_instances, len(CLASSES), args.d_model).cuda()
#     model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
#     matcher = HungarianMatcher(args.class_weight, args.giou_weight, args.l1_weight)
#     criterion = DetCriterion(args.no_instance_coef)
#
#     optimized_parameters = [
#         {'params': [p for n, p in model.module.named_parameters() if 'backbone' not in n and p.requires_grad]},
#         {'params': [p for n, p in model.module.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr / 10}
#     ]
#     optimizer = optim.AdamW(optimized_parameters, lr=args.lr, weight_decay=args.weight_decay)
#     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)
#
#     train_dataset = AugVocTrainDataset(args.dataset_root, args.input_size, args.input_size, args.num_instances)
#     train_sampler = data.distributed.DistributedSampler(train_dataset)
#     train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
#
#     val_dataset = AugVocValDataset(args.dataset_root, args.input_size, args.input_size)
#     val_sampler = data.distributed.DistributedSampler(val_dataset)
#     val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
#
#     for epoch_idx in range(args.epochs):
#         train_sampler.set_epoch(epoch_idx)
#         engine.train_one_epoch(model, optimizer, matcher, criterion, lr_scheduler, train_dataloader, dist_logger, epoch_idx)
#
#         val_sampler.set_epoch(epoch_idx)
#         engine.val_one_epoch(model, val_dataloader, val_dataset.coco, dist_logger, epoch_idx)


if __name__ == '__main__':
    # __main__()
    from model.polar_inst import PolarInst

    num_polars = 36
    num_channels = 256
    num_classes = 21

    model = PolarInst(num_polars, num_channels, num_classes).cuda()
    inp = torch.rand(2, 3, 1280, 768).cuda()
    out = model(inp)

    flatten_cls = [head_out['cls'].permute(0, 2, 3, 1).reshape(-1, num_classes) for head_out in out.values()]
    flatten_bbox = [head_out['bbox'].permute(0, 2, 3, 1).reshape(-1, 4) for head_out in out.values()]
    flatten_centerness = [head_out['centerness'].permute(0, 2, 3, 1).reshape(-1) for head_out in out.values()]

    flatten_cls = torch.cat(flatten_cls)
    flatten_bbox = torch.cat(flatten_bbox)
    flatten_centerness = torch.cat(flatten_centerness)

    print(flatten_cls.shape, flatten_bbox.shape, flatten_centerness.shape)
