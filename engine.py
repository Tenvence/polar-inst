import torch
import torch.nn.utils
import torch.cuda.amp as amp
import torchvision.ops as cv_ops

import utils.bbox_ops as bbox_ops


def train_one_epoch(model, optimizer, criterion, lr_scheduler, data_loader, dist_logger, epoch_idx):
    losses, cls_losses, bbox_losses, centerness_losses = [], [], [], []

    model.train()
    scaler = amp.GradScaler()
    processor = dist_logger.init_processor(data_loader)
    for img, data in processor:
        img = img.cuda(non_blocking=True)
        class_targets = data['class_targets'].cuda(non_blocking=True)
        bbox_targets = data['distance_targets'].cuda(non_blocking=True)
        centerness_targets = data['centerness_targets'].cuda(non_blocking=True)
        points = data['points'].cuda(non_blocking=True)

        with amp.autocast():
            class_pred, distance_pred, centerness_pred = model(img)
            loss_cls, loss_bbox, loss_centerness = criterion(
                {'class': class_pred, 'distance': distance_pred, 'centerness': centerness_pred},
                {'class': class_targets, 'distance': bbox_targets, 'centerness': centerness_targets},
                points
            )
            loss = loss_cls + loss_bbox + loss_centerness

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.clone().detach())
        cls_losses.append(loss_cls.clone().detach())
        bbox_losses.append(loss_bbox.clone().detach())
        centerness_losses.append(loss_centerness.clone().detach())

        cur_loss = dist_logger.reduce_tensor(loss)
        avg_loss = dist_logger.reduce_epoch_loss(losses)
        dist_logger.update_processor(processor, f'Epoch: {epoch_idx + 1}, avg_loss: {avg_loss:.2f}, cur_loss: {cur_loss:.2f}')

        lr_scheduler.step()

    dist_logger.update_tensorboard(super_tag='loss', tag_scaler_dict={
        'loss': dist_logger.reduce_epoch_loss(losses),
        'cls': dist_logger.reduce_epoch_loss(cls_losses),
        'bbox': dist_logger.reduce_epoch_loss(bbox_losses),
        'centerness': dist_logger.reduce_epoch_loss(centerness_losses)
    }, idx=epoch_idx)
    dist_logger.save_model(model)


@torch.no_grad()
def val_one_epoch(model, data_loader, coco_gt, dist_logger, epoch_idx, nms_cfg):
    pred_instances = []
    nms_pre, cls_score_thr, iou_thr = nms_cfg['nms_pre'], nms_cfg['cls_score_thr'], nms_cfg['iou_thr']

    model.eval()
    processor = dist_logger.init_processor(data_loader)
    for img, data in processor:
        img = img.cuda(non_blocking=True)
        points = data['points'].cuda(non_blocking=True)
        img_info_list = coco_gt.loadImgs(data['img_id'].numpy())

        class_pred, distance_pred, centerness_pred = model(img)

        class_pred = class_pred.sigmoid()  # [B, num_points, num_classes]
        cls_pred_scores, cls_pred_indexes = class_pred.max(dim=-1)  # [B, num_points]
        bbox_pred = bbox_ops.convert_distance_to_bbox(points, distance_pred)  # [B, num_points, 4]
        centerness_pred = centerness_pred.sigmoid()  # [B, num_points]

        batch_size, _, num_classes = class_pred.shape
        _, _, ih, iw = img.shape

        for batch_idx in range(batch_size):
            b_cls_pred_scores, b_cls_pred_indexes, b_centerness_pred = cls_pred_scores[batch_idx], cls_pred_indexes[batch_idx], centerness_pred[batch_idx]  # [num_points]
            b_bbox_pred = bbox_pred[batch_idx, :]  # [num_points, 4]

            _, top_idx = (b_cls_pred_scores * b_centerness_pred).topk(nms_pre)  # [topk]

            top_class_pred_scores, top_class_pred_indexes, top_centerness_pred = b_cls_pred_scores[top_idx], b_cls_pred_indexes[top_idx], b_centerness_pred[top_idx]  # [topk]
            nms_scores = top_class_pred_scores * top_centerness_pred  # [topk]

            top_bbox_pred = b_bbox_pred[top_idx, :]  # [topk, 4]
            top_bbox_pred = cv_ops.clip_boxes_to_image(top_bbox_pred, size=(ih, iw))

            valid_mask = top_class_pred_scores > cls_score_thr
            valid_class_pred_scores, valid_class_pred_indexes, valid_nms_scores = top_class_pred_scores[valid_mask], top_class_pred_indexes[valid_mask], nms_scores[valid_mask]
            valid_bbox_pred = top_bbox_pred[valid_mask, :]

            keep_idx = cv_ops.batched_nms(valid_bbox_pred, valid_nms_scores, valid_class_pred_indexes, iou_thr)
            keep_class_pred_scores, keep_class_pred_indexes = valid_class_pred_scores[keep_idx], valid_class_pred_indexes[keep_idx]
            keep_bbox_pred = valid_bbox_pred[keep_idx, :]

            oh, ow = img_info_list[batch_idx]['height'], img_info_list[batch_idx]['width']
            keep_bbox_pred = bbox_ops.recover_bboxes(keep_bbox_pred, oh, ow, ih, iw)
            keep_bbox_pred = cv_ops.box_convert(keep_bbox_pred, in_fmt='xyxy', out_fmt='xywh')

            for cls_score, cls_idx, bbox in zip(keep_class_pred_scores, keep_class_pred_indexes, keep_bbox_pred):
                pred_instances.append({
                    'image_id': int(data['img_id'][batch_idx]),
                    'category_id': int(cls_idx) + 1,
                    'bbox': [float(str('%.1f' % coord)) for coord in bbox.tolist()],
                    'score': float(str('%.1f' % cls_score))
                })

    dist_logger.save_pred_instances_local_rank(pred_instances)
    dist_logger.save_val_file()
    dist_logger.update_tensorboard_val_results(coco_gt, epoch_idx)
