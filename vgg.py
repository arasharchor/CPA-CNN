import os
import h5py

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
import numpy as np

img_width = 96
img_height = 96
nb_classes = 20

def VGG_16(weights_path=None):
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))    
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))     
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))     
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))     
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    if weights_path:
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        #print f.attrs['nb_layers']
        for k in range(f.attrs['nb_layers'] - 6):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    return model

# Test pretrained model
#model = VGG_16('vgg16_weights.h5')
model = VGG_16()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy')

import glob
from PIL import Image
import numpy as np

X = [] #data
y = [] #labels

paths = glob.glob('./training_set/*')
for path in paths:
    label = path.split('/')[2]
    if label != 'interphase' and label != 'lines' and label != 'blurry':
        imgs = glob.glob(path + '/*')
        for img in imgs:
            im = Image.open(img)
            X += [np.array(im)]
            y += [label]

X = np.array(X)
y = np.array(y)

X = X.astype('float32')
X = X.reshape(X.shape[0], 3, img_width, img_height)

#Channel-wise mean and var
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std # Standardize
where_nans = np.isnan(X) # replace all NaNs ...
X[where_nans] = 0 # ... with 0
 
import sklearn
Xs, ys = sklearn.utils.shuffle(X,y,random_state=0)

from keras.utils import np_utils

# Convert labels to numeric
y_unique = np.unique(ys)
dic = {}

for i, label in enumerate(y_unique):
    dic[label] = i
print dic

y_numeric = []
for el in ys:
    y_numeric += [dic[el]]
    
y_numeric # now a 2000 label vector
Y = np_utils.to_categorical(y_numeric, nb_classes) # Categorical Y

# Create a image generator
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(Xs)

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)

model.fit_generator(datagen.flow(Xs, Y, batch_size=200),
                    samples_per_epoch=(len(Xs) * 10), nb_epoch=500, show_accuracy=True)

model.fit(Xs, Y, batch_size=200, nb_epoch=20000, validation_split=0.2, show_accuracy=True, callbacks=[checkpointer])

