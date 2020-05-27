# Object Detection with YOLOv3

## Purpose

This is a practice project based on my learning from Coursera's Andrew Ng's Convolutional Neural Network on car detection for Autonomous Driving using YOLO and Jason Brownlee's website on How to Perform Object Detection With YOLOv3 in Keras ("https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/")

## You Only Look Once (YOLO)
In YOLO, the input image dataset is provided to the Convolutional neural network (CNN) as a predifined set of grids (nxn). For each grid or cell, the target output (y) is composed of bounding boxes (bx, by, bh, bw) and classes of objects defined as a vector [Pc, bx, by, bh, bw, C1,...,Cn],

where:

Pc : probability that an object exists inside that cell (0-1)


bx, by = 2D cartesian coordinates of the object's center. (Object localization)

bw, bh = Width and height of the object, respectively. (Object localization)

C : Probability of object classes, C is more than 1 for multi object classification. 

Anchors: this paramter represents the number of boxes inside each cell. The example shown is only for one box (anchor) but if we have 2 anchors or more, then the same parameters are repeated twice  in this example inside each cell:

y = [Pc, bx, by, bh, bw, C1,..,Cm, Pc, bx, by, bh, bw, C1,..,Cm]

Loss Function:
Since this is a supervised learning problem, we have a ground true label y and the neural network output yhat. The loss function is the sum of the squared of the difference between y and yhat for all the elements inside the box vector, if an object exist. 

Shape of each cell: Let's say if an input image has 19x19 grid cells, with 3 classes and 2 anchors then the shape of y can be represented as (19 x 19 x 2 x 8), 8 is the total number of parameters inside each anchor [Pc, bx, by, bh, bw, C1, C2, C3].

Scoring: YOLO uses the highest score (and higher than a given threshold) for each box of each cell, by selecting the index of maximum probability of Pc AND class probabilities (C1, C2, C3). 

Example, if Pc = 0.6, and probability of each class are C1 = 0.1, C2 = 0.05, C3 = 0.15, then the product of Pc * C3 is the maximum value between the three. Argmax is then used to select the predifined colors or names to that class number 3.

### Non-Max Supppression
This method is used to get rid of boxes with low scores (less than a provided threshold). It also enables selection of only 1 box when several boxes overlap by detecting the same object.


## Keras-YOLO3 (experiencor)
“keras-yolo3: Training and Detecting Objects with YOLO3” by Huynh Ngoc Anh or experiencor is a pre-trained single python file "yolo3-one-file-to-detect-them-all.py" designed to use YOLO with Keras. It provides scripts to both load and use pre-trained YOLO 3 as well as transfer learning weights for developing YOLOv3 models on new datasets.

### Make the model (Yolo3_tranfer_learning.py)

Yolo3_tranfer_learning.py uses the function "make_yolov3_model" from "yolo3_one_file_to_detect_them_all" to first create the model. Next, to instantiate the weights, we have to first download the "YOLOv3 Pre-trained Model Weights (yolov3.weights) (237 MB)" from the website into the working directory. The class "WeightReader" from "yolo3_one_file_to_detect_them_all.py" file is instantiated with the path to the downloaded weights file. This will parse the file and load the model weights into memory in a format that we can set into our Keras model. Keras helper function _conv_block inside the "make_yolov3_model" builds the convolutional layers, filters, activation functions, padding, strikes and ther parameters and hypermarameters in better optimizing the model. The model is then saved into model.h5 for further prediction and image classificaiton in the next step.

### Model prediction (Yolo3_predict.py)

We can predict an image using a new photo for object detection from a set of provided images that the model knows about from the "MSCOCO dataset", zebra.jpg and dog.jpg are good examples.

Yolo3_predict.py uses the functions "load_img" and "img_to_array" from keras.preprocessing.image to load model.h5 and predict yhat. yhat is NumPy arrays that defines the probability  of object's bounding boxes and class labels but are encoded. We can output both the yhat and the shape of the arrays. They must be interpreted in the next file. 

### Model Interpretation (Yolo3_interpret.py)

Uses decode_netout function from experiencor to decode the predicted yhat arrays output. A threshold of 0.6 is used to select the highest probability between the element in each box. The function do_nms() is used for non-max-supperssion with 0.5 as threshold.

correct_yolo_boxes() function  performs the translation of bounding box coordinates, taking the list of bounding boxes, the original shape of our loaded photograph, and the shape of the input to the network as arguments. The coordinates of the bounding boxes are updated directly.


![](Dog.png)


