import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import argparse

classes = ['Not a Meme', 'Meme']

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--image_path', type = str, help ='image path for inference')
parser.add_argument('-model', '--classification_model', type = str, help ='model to load', default='models/modelv2.h5') 
 
args = parser.parse_args()

model = load_model('modelv2.h5', compile=False)

path = args.image_path

img = cv2.imread(path)
img_copy = cv2.resize(img, (224, 224))
img_copy = img_copy.reshape(1, 224, 224, 3)
preds = np.argmax(model.predict(img_copy))
preds = classes[preds]
print('Image is ', preds)

