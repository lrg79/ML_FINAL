import os
import sys
import pickle
import time

import numpy as np

import keras

from sklearn import preprocessing

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

# DATA LOADING


le = preprocessing.LabelEncoder()

training_data = np.load('cs4780sp2018finalproject/train-resnet-features.npy').item()
labels = np.concatenate(np.vstack([i.split('/')[1] for i in training_data.keys()]))

classes = np.unique(labels)

le.fit(classes)
labels = le.transform(labels)

y_train = keras.utils.to_categorical(labels, num_classes=classes.shape[0])
# y_train = labels
x_train = np.vstack(list(training_data.values()))

n,d = x_train.shape

idx = np.random.permutation(n)
x_train = x_train[idx]
y_train = y_train

test_data = np.load('cs4780sp2018finalproject/val-resnet-features.npy').item()
file_names = np.concatenate(np.vstack(list(test_data.keys())))
xTe = np.vstack(list(test_data.values()))

# TODO: test splits

# DEFINE MODEL

model = Sequential()

model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(classes.shape[0], activation='softmax'))

sgd = optimizers.SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# TRAINING
model.fit(x_train, y_train, epochs=200, batch_size=256)

# PREDS
preds = model.predict(xTe, batch_size=128, verbose=1)


pred_labels = np.argmax(preds, axis=1)
names = le.inverse_transform(pred_labels)

file = open('preds.csv', 'w+')
file.write("image_label,celebrity_name\n")
file.close()
file = open('preds.csv', 'a')
np.savetxt(file, np.column_stack((file_names, names)), fmt='%s', delimiter=',')
file.close()

