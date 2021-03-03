import pycocotools.cocoeval as coco_eval

import utils.segm_coco_evaluate


def bbox_evaluate(coco_gt, res_file):
    coco_evaluator = coco_eval.COCOeval(cocoGt=coco_gt, cocoDt=coco_gt.loadRes(res_file), iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.stats


def segm_evaluate(coco_gt, res_file):
    coco_evaluator = utils.segm_coco_evaluate.SegCocoEval(coco_gt, coco_gt.loadRes(res_file))
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    ap25, ap50, ap70, ap75, abo = coco_evaluator.stats
    return ap25, ap50, ap70, ap75, abo
