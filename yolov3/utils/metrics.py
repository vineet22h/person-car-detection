import numpy as np

def get_map(gts, preds, num_classes, iou_thresh):
    aps = []
    counts = []
    for class_idx in range(num_classes):
        class_gts, class_preds, npos = parse_gt_pred_data(gts, preds, class_idx)
        rec, prec, ap, count = get_ap(class_gts, class_preds, npos, iou_thresh=iou_thresh, use_07_metric=False)
        aps.append(ap)
        counts.append(count)

    aps = np.array(aps)
    counts = np.array(counts)
    mAP = np.sum(aps * counts) / np.sum(counts)
    return mAP

def parse_gt_pred_data(gts, preds, class_idx):
    npos = 0
    class_gts = []
    for i in range(len(gts)):
        bboxes = []
        is_crowds = []
        if len(gts[i]['bboxes']) > 0:
            for (gt_box, gt_class, is_crowd) in zip(gts[i]['bboxes'], gts[i]['classes'], gts[i]['is_crowds']):
                if gt_class == class_idx:
                    bboxes.append(gt_box)
                    is_crowds.append(is_crowd)
            is_crowds = np.array(is_crowds).astype(np.bool)
            npos+=sum(~is_crowds)
            used = [False]*len(bboxes)
            class_gts.append({'bbox': bboxes, 'is_crowd': is_crowds, 'used': used})
        else:
            class_gts.append({'bbox': [], 'is_crowd': [], 'used': []})
        
    class_preds = []
    for i in range(len(gts)):
        bboxes = []
        scores = []
        if type(preds[i]['bboxes']) == np.ndarray:
            for (pred_box, pred_class, pred_score) in zip(preds[i]['bboxes'], preds[i]['classes'], preds[i]['scores']):
                if pred_class == class_idx:
                    bboxes.append(pred_box)
                    scores.append(pred_score)
            class_preds.append({'bbox': bboxes, 'scores': scores})
        else:
            class_preds.append({'bbox': [], 'scores': []})
    
    return class_gts, class_preds, npos

def get_ap(class_gts, class_preds, npos, iou_thresh = 0.5, use_07_metric=False):
        """
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        
        nd = len(class_gts)  # number of predict data
        tp = [0] * nd
        fp = [0] * nd
        tp_count = 0
                 
        for idx, (gt, pred) in enumerate(zip(class_gts, class_preds)):
            pred_bboxes = [pred_boxes for pred_boxes in pred['bbox']]
            max_iou = 0.0
            max_index = -1
            for j in range(len(pred['bbox'])):
                for i in range(len(gt['bbox'])) :
#                     print(pred['bbox'][j], gt['bbox'][i])
                    iou = box_iou(pred['bbox'][j], gt['bbox'][i])

                    if iou > max_iou and gt['used'][i] == 0:
                        max_iou = iou
                        max_index = i

                # drop the prediction if couldn't match iou threshold
                if max_iou < iou_thresh:
                    max_index = -1

                if max_index != -1:
                    gt['used'][max_index] = 1
                    tp[idx] = 1
                    tp_count += 1
                else:
                    fp[idx] = 1
#             print(tp, fp)
        
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
    
        rec = tp / float(npos+np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        return rec, prec, ap, nd 

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    inter_w = max(0.0, inter_box[2] - inter_box[0] + 1)
    inter_h = max(0.0, inter_box[3] - inter_box[1] + 1)

    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_area = inter_w * inter_h
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)