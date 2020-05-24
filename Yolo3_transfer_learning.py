import struct
import numpy as np
from keras.models import Model
from yolo3_one_file_to_detect_them_all import make_yolov3_model 
from yolo3_one_file_to_detect_them_all import WeightReader

# define the model
model = make_yolov3_model()
# load the model weights
weight_reader = WeightReader('yolov3.weights')
# set the model weights into the model
weight_reader.load_weights(model)
# save the model to file
model.save('model.h5')
