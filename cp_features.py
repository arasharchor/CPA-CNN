#!/usr/bin/env python


import pandas as pd
df = pd.read_csv('Anne_DNA_66.csv')
df = df[df['Class'] != 'interphase']

y = df['Class'].values

df = df.drop('TableNumber', 1)
df = df.drop('ImageNumber', 1)
df = df.drop('ObjectNumber', 1)
df = df.drop('Class', 1)
df = df.drop('Nuclei_AreaShape_EulerNumber', 1)

df_norm = (df - df.mean()) / (df.max() - df.min())
df_norm.values.shape

# Convert y into Y
nb_classes = 23
import numpy as np
from keras.utils import np_utils, generic_utils

# Convert labels to numeric
y_unique = np.unique(y)
dic = {}

for i, label in enumerate(y_unique):
    dic[label] = i
print dic

y_numeric = []
for el in y:
    y_numeric += [dic[el]]
    
y_numeric # now a 2000 label vector
Y = np_utils.to_categorical(y_numeric, nb_classes)

print Y.shape

Y_train = Y

X_train = df_norm.values

from sklearn.cross_validation import StratifiedKFold

#print y_numeric
skl = StratifiedKFold(y_numeric, n_folds=5)


for train,test in skl:
    print len(train), len(test)
    
X_stratified = X_train[train]
Y_stratified = Y[train]

X_stratified = np.append(X_stratified,X_train[test],axis=0)
Y_stratified = np.append(Y_stratified,Y[test],axis=0)

X_stratified.shape
Y_stratified.shape

## Convolutional Neural Network with 2 convolutions

from keras.models import Sequential
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.layers.normalization import BatchNormalization

nb_epoch = 12

### Network

model = Sequential()
model.add(Dense(64, input_dim=97))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4069))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(X_stratified, Y_stratified, nb_epoch=100000,validation_split=0.2, show_accuracy=True)






