from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np

# import your data here instead
# X - inputs, 10000 samples of 128-dimensional vectors
# y - labels, 10000 samples of scalars from the set {0, 1, 2}

X = np.random.rand(10000, 128).astype("float32")
y = np.random.randint(3, size=(10000,1))

# process the data to fit in a keras CNN properly
# input data needs to be (N, C, X, Y) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size

X = X.reshape((10000, 1, 128, 1))

# output labels should be one-hot vectors - ie,
# 0 -> [0, 0, 1]
# 1 -> [0, 1, 0]
# 2 -> [1, 0, 0]
# this operation changes the shape of y from (10000,1) to (10000, 3)

y = np_utils.to_categorical(y)

# define a CNN
# see http://keras.io for API reference

cnn = Sequential()
cnn.add(Convolution2D(64, (3,1), 1,
    border_mode="same",
    activation="relu",
    input_shape=(1, 128, 1)))
cnn.add(Convolution2D(64, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1)))

cnn.add(Convolution2D(128, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(128, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(128, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1)))
    
cnn.add(Convolution2D(256, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(256, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(256, (3,1), 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1)))
    
cnn.add(Flatten())
cnn.add(Dense(1024, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation="softmax"))

# define optimizer and objective, compile cnn

cnn.compile(loss="categorical_crossentropy", optimizer="adam")

# train

cnn.fit(X, y, nb_epoch=20, show_accuracy=True)