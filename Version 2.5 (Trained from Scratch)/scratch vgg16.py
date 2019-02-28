
# coding: utf-8

# In[ ]:


# Importing Dependencies

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

from keras import applications
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import EarlyStopping
import matplotlib as plt

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

import time
import sys
sys.path.append('../')
from Utilities.model_visualization import model_to_png

print('Imports Done')


# In[ ]:


#make model

def vgg16():
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(150,150,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    return model


# In[ ]:


# paths to training and validation data

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/test'


# In[ ]:


# Params for CNN

img_width, img_height = 150, 150
batch_size = 5
epochs = 200
train_samples = 420
validation_samples = 80


# In[ ]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
train_batchsize = 210
val_batchsize = 40
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=train_batchsize,
        class_mode='categorical') # class_mode='categorical'
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='categorical', # class_mode='categorical'
        shuffle=False)


# In[33]:


#Compile the model
model = vgg16()
model.compile(loss='binary_crossentropy', #categorical_crossentropy
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# define the checkpoint
filepath = "model_finetuned.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=8, verbose=1, mode='max')
callbacks_list = [checkpoint, early_stopping]

start = time.time()

#Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=60,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks=callbacks_list)

end = time.time()


# In[ ]:


model.save_weights('weights.h5')

print(start-end);
# In[ ]:


prediction_data_dir = 'dataset/predict'
# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
        prediction_data_dir,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

