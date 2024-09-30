import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import keras

from keras.preprocessing.image import  ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

#EDA
len(os.listdir("/content/drive/MyDrive/data/dataset/train"))

train_datagen= ImageDataGenerator(zoom_range=0.5,shear_range= 0.3 ,horizontal_flip=True,preprocessing_function= preprocess_input)
val_datagen=ImageDataGenerator(preprocessing_function= preprocess_input)
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)


train=train_datagen.flow_from_directory(directory="/content/drive/MyDrive/data/dataset/train",target_size=(256,256),batch_size=32)
val=val_datagen.flow_from_directory(directory="/content/drive/MyDrive/data/dataset/validation",target_size=(256,256),batch_size=32)
test=test_datagen.flow_from_directory(directory="/content/drive/MyDrive/data/dataset/test",target_size=(256,256),batch_size=32)

t_img,label = train.next()
def plotImage(img_arr,label):

  for im , l in zip(img_arr , label):
    plt.figure(figsize=(5,5))
    plt.show()


plotImage(t_img[:3],label[:3])
