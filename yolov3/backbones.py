import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *

from yolo_v3.layers import *

def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    ## block : 1
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)
    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    ## block : 2
    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    ## block : 3
    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    ## block : 4
    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)
    
    ## block : 5
    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data