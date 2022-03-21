import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo_v3.utils.config import *
from yolo_v3.losses import yolo_loss
from yolo_v3.models.yolov3 import YOLOv3
from yolo_v3.utils.metrics import get_map
from yolo_v3.generator import BatchGenerator
from yolo_v3.create_models import create_yolov3
from yolo_v3.utils.helper import resize, predict, show_img

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class YoloV3():
    def __init__(self, model_weight_path = '', training = True):
        self.training = training
        self.net_h = NET_H
        self.net_w = NET_W
        self.num_classes = NUM_CLASSES
        self.model_weight_path = model_weight_path
        self.build_model()
    
    def build_model(self):
        if self.training:
            self.train_model = create_yolov3(input_shape = (self.net_h, self.net_w, 3), load_pretrained = False)
            self.train_model.compile(optimizer=Adam(lr=1e-3), 
                                loss= {'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            input_layer  = Input((self.net_h, self.net_w, 3))
            output_layer = YOLOv3(input_layer, self.num_classes)
            self.inference_model = Model(input_layer, output_layer)
            self.inference_model.load_weights(self.model_weight_path)
                    
    def train(self, train_inst, val_inst, base_path, batch_size, checkpoint_path):
        ##Generator
        train_generator = BatchGenerator(train_inst, base_path, batch_size = batch_size, phase = 'train')
        valid_generator = BatchGenerator(val_inst, base_path, batch_size = 1, phase = 'val')
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only = True, verbose = 1),
            ReduceLROnPlateau(patience = 3, verbose = 1, factor = 0.1),
            EarlyStopping(verbose = 1, patience = 5)]
        
        ##Fit model
        self.train_model.fit(x = train_generator,
                            steps_per_epoch = len(train_inst)//BATCH_SIZE,
                            verbose = 1,
                            epochs = 100,
                            validation_data = valid_generator,
                            validation_steps = len(val_inst),
                            callbacks = callbacks)
        
    def predict_img(self, img_path, show = True):
        img = cv2.imread(img_path)
        img, _ = resize(img, [], self.net_h, self.net_w)
        bboxes, scores, classes = predict(self.inference_model, img/255.0)
        if show:
            show_img(img, bboxes, classes, ['person', 'car'], scores = scores, add_scores = True)
        return bboxes, scores, classes
    
    def evaluate(self, test_inst, base_path, iou_thresh):
        gts = []
        preds = []
        for data_pt in tqdm(test_inst):
            bboxes = [obj['bbox'] for obj in data_pt['bboxes']]
            is_crowd = [obj['iscrowd'] for obj in data_pt['bboxes']]
            classes = [obj['category_id'] for obj in data_pt['bboxes']]            
            pred_bbox, pred_scores, pred_classes = self.predict_img(f"{base_path}/{data_pt['file_name']}", show = False)
        
            preds.append({'bboxes': pred_bbox, 'scores': pred_scores, 'classes': pred_classes})
            gts.append({'bboxes': bboxes, 'is_crowds': is_crowd, 'classes': classes})
        map_ = get_map(gts, preds, self.num_classes, iou_thresh)
        return map_