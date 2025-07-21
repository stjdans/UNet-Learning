import tensorflow as tf
import numpy as np
import keras

from keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose, CenterCrop, Concatenate
from keras.models import Model

def build_unet_model(input_shape, num_class=2):
   inputs = keras.Input(shape=input_shape)
   
	# Contracting path (downsampling)
   enc1_1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
   enc1_2 = Conv2D(64, 3, activation='relu', padding='same')(enc1_1)
   pool1 = MaxPooling2D(2)(enc1_2)

   enc2_1 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
   enc2_2 = Conv2D(128, 3, activation='relu', padding='same')(enc2_1)
   pool2 = MaxPooling2D(2)(enc2_2)

   enc3_1 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
   enc3_2 = Conv2D(256, 3, activation='relu', padding='same')(enc3_1)
   pool3 = MaxPooling2D(2)(enc3_2)

   enc4_1 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
   enc4_2 = Conv2D(512, 3, activation='relu', padding='same')(enc4_1)
   pool4 = MaxPooling2D(2)(enc4_2)

	# Bottleneck
   enc5_1 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
   dec5_1 = Conv2D(1024, 3, activation='relu', padding='same')(enc5_1)

	# Expansive path (upsampling)
   uppool4 = Conv2DTranspose(512, 2, strides=2, activation='relu', padding='same')(dec5_1)
   crop4 = CenterCrop(uppool4.shape[1], uppool4.shape[1])(enc4_2)
   concat4 = Concatenate(axis=-1)([uppool4, crop4])
   dec4_2 = Conv2D(512, 3, activation='relu', padding='same')(concat4)
   dec4_1 = Conv2D(512, 3, activation='relu', padding='same')(dec4_2)

   uppool3 = Conv2DTranspose(256, 2, strides=2, activation='relu', padding='same')(dec4_1)
   crop3 = CenterCrop(uppool3.shape[1], uppool3.shape[1])(enc3_2)
   concat3 = Concatenate(axis=-1)([uppool3, crop3])
   dec3_2 = Conv2D(256, 3, activation='relu', padding='same')(concat3)
   dec3_1 = Conv2D(256, 3, activation='relu', padding='same')(dec3_2)

   uppool2 = Conv2DTranspose(128, 2, strides=2, activation='relu', padding='same')(dec3_1)
   crop2 = CenterCrop(uppool2.shape[1], uppool2.shape[1])(enc2_2)
   concat2 = Concatenate(axis=-1)([uppool2, crop2])
   dec2_2 = Conv2D(128, 3, activation='relu', padding='same')(concat2)
   dec2_1 = Conv2D(128, 3, activation='relu', padding='same')(dec2_2)

   uppool1 = Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')(dec2_1)
   crop1 = CenterCrop(uppool1.shape[1], uppool1.shape[1])(enc1_2)
   concat1 = Concatenate(axis=-1)([uppool1, crop1])
   dec1_2 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
   dec1_1 = Conv2D(64, 3, activation='relu', padding='same')(dec1_2)

   outputs = Conv2D(num_class, 1, activation='softmax' if num_class >= 2 else 'sigmoid')(dec1_1)
   
   return Model(inputs, outputs)