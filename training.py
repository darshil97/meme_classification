import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet import preprocess_input


model = DenseNet169(weights='imagenet',
                              include_top=False,
                              input_shape=(224,224,3))


img_width, img_height = 224, 224
train_data_dir = 'dataset/train'
train_images = 1350*2
val_data_dir = 'dataset/test'
val_images = 450*2
batch_size = 10
epochs = 10


train_datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True)

val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='categorical')


for layer in model.layers:
    layer.trainable = True


x = model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

final_mod = Model(inputs=model.input, outputs=output)
adam = Adam(learning_rate=0.000001)
final_mod.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

data = final_mod.fit(train_generator,
                          steps_per_epoch=train_images//batch_size,
                          epochs=epochs,
                          shuffle=True,
                          validation_data=val_generator, validation_steps=val_images//batch_size)


final_mod.save('modelv2.h5') 




