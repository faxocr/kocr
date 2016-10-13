import numpy as np
np.random.seed(71)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


nb_dim = 48
batch_size = 128
nb_epoch = 200


def load_data():
    X, labels = np.load("images.npy"), np.load("labels.npy")
    X = X.astype(np.float32).reshape([-1, 1, nb_dim, nb_dim]) / 255
    y = np.zeros(labels.shape)
    unique_label = np.unique(labels)
    for i, label in enumerate(unique_label):
        y[labels == label] = i
    y = np_utils.to_categorical(y.astype(np.uint8))
    return X, y, unique_label


print "Load data"
X, y, unique_label = load_data()
n = X.shape[0]


print "Split data into train set and validation set"
idx = np.random.permutation(n)
n_train = int(n * 0.9) / batch_size * batch_size
X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
X_valid, y_valid = X[idx[n_train:]], y[idx[n_train:]]
samples_per_epoch = 10 * n_train


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

datagen = ImageDataGenerator(
    zoom_range=[0.9, 1.05],
    rotation_range=20)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    samples_per_epoch=samples_per_epoch,
                    validation_data=(X_valid, y_valid),
                    callbacks=[earlystopping, checkpointer],
                    nb_epoch=nb_epoch, verbose=1)

model.load_weights("weights.hdf5")
print "Testing on keras:", (model.predict_classes(X) == y.argmax(axis=1)).mean()


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
