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

# data = np.load('train-resnet-features.npy').item()
# labels = np.concatenate(np.vstack([i.split('/')[1] for i in data.keys()]))
# features = np.vstack(list(data.values()))
# classes = np.unique(labels)

train_data_dir = "images-train"
val_data_dir = "images-val-pub"

if K.image_data_format() == 'channels_first':
	input_shape = (3, 120, 120)
else:
	input_shape = (120, 120, 3)

model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(98))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
			  optimizer='Nadam',
			  metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
									train_data_dir,
									target_size=(120, 120),
									batch_size=32,
									class_mode="categorical")
model.fit_generator(
		train_generator,
		epochs=10)

# model.save("idk.h5")

file = open("preds.csv", "w")

for filename in os.listdir(val_data_dir):
	img = cv2.imread(os.path.join(val_data_dir, filename))
	pred = model.predict(img)
	file.write(str(filename) + "," + pred)

file.close()

