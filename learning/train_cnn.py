import numpy as np
np.random.seed(71)
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam


nb_dim = 48
batch_size = 128
nb_epoch = 100


def load_data():
    X, labels = np.load("images.npy"), np.load("labels.npy")
    X = X.astype(np.float32).reshape([-1, 1, nb_dim, nb_dim]) / 255
    y = np.zeros(labels.shape)
    unique_label = np.unique(labels)
    for i, label in enumerate(unique_label):
        y[labels == label] = i
    y = np_utils.to_categorical(y.astype(np.uint8))
    return X, y, unique_label


def augment_data(X, y):
    X_list, y_list = [], []
    for i in xrange(X.shape[0]):
        img_i = X[i][0]
        X_list.append(img_i)
        y_list.append(y[i])
        for _ in range(10):
            angle = np.random.uniform(-30, 30)
            scale = np.random.uniform(0.9, 1.05)
            rotation_matrix = cv2.getRotationMatrix2D((24, 24), angle, scale)
            img_rot = cv2.warpAffine(img_i, rotation_matrix, img_i.shape,
                                     flags=cv2.INTER_LINEAR)
            X_list.append(img_rot)
            y_list.append(y[i])
    return np.array(X_list).reshape([-1, 1, 48, 48]), np.array(y_list)


print "Load data"
X, y, unique_label = load_data()


print "Augment data"
X, y = augment_data(X, y)
idx = np.random.permutation(X.shape[0])
X, y = X[idx], y[idx]


print "Build model"
model = Sequential()

model.add(Convolution2D(32, 9, 9,
                        border_mode='valid',
                        input_shape=(1, nb_dim, nb_dim)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(unique_label)))
model.add(Activation('softmax'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=["accuracy"])


print "Fit"
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                               verbose=0, save_best_only=True)
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
          verbose=1, validation_split=0.1,
          callbacks=[earlystopping, checkpointer])
model.load_weights("weights.hdf5")


def dump_weights(filename, model):
    with open(filename, "w") as fp:
        fp.write("{} ".format(len(unique_label)))
        for layer in model.layers:
            for w in layer.get_weights():
                for v in w.reshape([-1]):
                    fp.write("{} ".format(v))
        for label in unique_label:
            fp.write("{} ".format(label))

print "Dump results to txt"
dump_weights("cnn-result.txt", model)
