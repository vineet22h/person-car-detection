NUM_CLASSES             = 2

## MODEL
BATCH_SIZE              = 4
NET_H                   = 416
NET_W                   = 416
ANCHOR_PER_SCALE        = 3
STRIDES                 = [8, 16, 32]
MAX_BBOX_PER_SCALE      = 100
FREEZE_DARKNET          = False
IOU_LOSS_THRESH              = 0.7
# YOLO_TYPE               = "yolov3"
# if YOLO_TYPE                == "yolov4":
#     YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
#                                [[36,  75], [76,   55], [72,  146]],
#                                [[142,110], [192, 243], [459, 401]]]
# if YOLO_TYPE                == "yolov3":
YOLO_ANCHORS            = [[[16, 27], [27, 73], [44, 139]],
                            [[91, 69], [72, 242], [175, 141]],
                            [[159, 339], [323, 222], [371, 361]]]

# YOLO_ANCHORS            = [[[27, 16], [73, 27], [139, 44]],
#                            [[69, 91], [242, 72], [141, 175]],
#                            [[339, 159], [222, 323], [361, 371]]]

MASKS                   = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

## LOGS and Checkpoint
# CHECKPOINT_PATH = './checkpoint/yolov3_1.h5'
# PRETRAINED_WEIGHTS = './checkpoint/yolov3.weights'

## TEST
OBJECT_THRESH           = 0.35
NMS_THRESH              = 0.1