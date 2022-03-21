import cv2
import numpy as np
import tensorflow as tf
from albumentations import *
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolo_v3.utils.config import *

def show_img(img, pred_bboxes = None, pred_classes = None, class_names = None, add_scores = False, scores = None, save = False, img_name = 'saved_img.jpeg'):
    fig, ax = plt.subplots(1, figsize = (50, 50))  
    colors = ['red', 'blue']
    if type(pred_bboxes) == np.ndarray:
        for i in range(len(pred_bboxes)):
            rect1 = patches.Rectangle(
                (pred_bboxes[i][0], pred_bboxes[i][1]),
                pred_bboxes[i][2], 
                pred_bboxes[i][3], 
                linewidth= 4, 
                edgecolor=colors[pred_classes[i]], 
                facecolor='none')
            ax.add_patch(rect1)
        
            if add_scores:
                rect2 = patches.Rectangle(
                    (pred_bboxes[i][0], pred_bboxes[i][1]-7),
                    40,
                    7,
                    linewidth=4, 
                    edgecolor=colors[pred_classes[i]],
                    facecolor = colors[pred_classes[i]])
                ax.add_patch(rect2)
                class_name = class_names[pred_classes[i]]
                score = str(scores[i])[:4]
                ax.text(pred_bboxes[i][0], pred_bboxes[i][1], f"{class_name}: {score}", fontsize = 40, fontdict = {'color': 'white'})
      
    ax.imshow(img)
    if save:
        fig.savefig(img_name)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def resize(img, bboxes, net_h, net_w):
       labels = np.ones((len(bboxes), ))
       aug = Compose([
           Resize(net_h, net_w), 
           ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
       
       aug_result = aug(image=img, bboxes=bboxes, labels = labels)
       
       return aug_result['image'], aug_result['bboxes']
   
def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))    

    
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(b1_y2, b2_y2)                    
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def scale_to_bbox(scale, anchors, mask):
    grid_h, grid_w, num_boxes = map(int, scale.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)
    
    # Reshape to batch, height, width, num_anchors, box_params.
    scale = scale[0]
    box_xy = sigmoid(scale[..., :2])
    box_wh = np.exp(scale[..., 2:4])
    box_wh = box_wh * anchors_tensor
    
    box_confidence = sigmoid(scale[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(scale[..., 5:])
    
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (NET_H, NET_W)
    box_xy -= (box_wh / 2.)   
    boxes = np.concatenate((box_xy, box_wh), axis=-1)
    
    box_scores = box_confidence * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJECT_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores
    
def post_process_bbox(scale_1, scale_2, scale_3):
    boxes, classes, scores = [], [], []
    for scale, mask in zip([scale_1, scale_2, scale_3], MASKS):
        b, c, s = scale_to_bbox(scale, np.array(YOLO_ANCHORS).reshape(-1, 2), mask)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    
    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    
    # Scale boxes back to original image shape.
    w, h = NET_W, NET_H
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    return boxes, scores, classes

def nms( boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep

def predict(model, img):
    pred = model.predict(tf.expand_dims(img, axis = 0))
    
    output_net_h = [NET_H//s for s in STRIDES]
    output_net_w = [NET_W//s for s in STRIDES]

    
    scale_1 = np.reshape(pred[2], (1, output_net_h[0], output_net_w[0],
                                   3, 5+NUM_CLASSES))
    scale_2 = np.reshape(pred[1], (1, output_net_h[1], output_net_w[1],
                                   3, 5+NUM_CLASSES))
    scale_3 = np.reshape(pred[0], (1, output_net_h[2], output_net_w[2],
                                   3, 5+NUM_CLASSES))
    
    bboxes, scores, classes = post_process_bbox(scale_1, scale_2, scale_3)
    
    return bboxes, scores, classes
