# person-car-detection using yolov3
The project is implementation of person car detection using yolov3.
YOLOv3 is extremely fast and accurate. In mAP measured at 0.5 IOU YOLOv3 is on par with Focal Loss but about 4x faster. Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model. The model uses three different scales for prediction which encounters the problem of different sizes of bounding box.
Inference is done using Mean Average Precision (MAP) as a metric.<br>

# Dataset <br>
https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz
