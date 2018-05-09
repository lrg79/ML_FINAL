import os
import sys
import pickle
import time

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

# data = np.load('train-resnet-features.npy').item()
# labels = np.concatenate(np.vstack([i.split('/')[1] for i in data.keys()]))
# features = np.vstack(list(data.values()))
# classes = np.unique(labels)

train_data_dir = "images-train"
val_data_dir = "images-val-pub"

img_width = 160
img_height = 160

batch_size = 32
epochs = 10

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(98))
model.add(BatchNormalization())
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
			  optimizer='Nadam',
			  metrics=['accuracy'])

train_datagen = ImageDataGenerator(
					rescale=1./255,
					horizontal_flip=True,
					fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
									train_data_dir,
									target_size=(img_width, img_height),
									batch_size=batch_size,
									class_mode="categorical")

classes = train_generator.class_indices
# print(classes)

model.fit_generator(
		train_generator,
		epochs=epochs)

model.save("kardashian3.h5")
