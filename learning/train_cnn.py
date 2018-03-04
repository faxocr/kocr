import numpy as np
np.random.seed(71)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import struct


def load_data():
    X, labels = np.load('num_images.npy'), np.load('num_labels.npy')
    X = X.astype(np.float32).reshape([-1, 1, nb_dim, nb_dim]) / 255
    y = np.zeros(labels.shape)
    unique_label = np.unique(labels)
    for i, label in enumerate(unique_label):
        y[labels == label] = i
    y = np_utils.to_categorical(y.astype(np.uint8))
    return X, y, unique_label


def build_model(nb_dim, nb_output):
    model = Sequential()

    model.add(Conv2D(32, (9, 9), activation='relu', padding='valid', input_shape=(1, nb_dim, nb_dim)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='valid', input_shape=(1, nb_dim, nb_dim)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid', input_shape=(1, nb_dim, nb_dim)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_output, activation='softmax'))
    return model


def dump_weights(filename, model, unique_label):
    b = bytearray()
    b += struct.pack('i', len(unique_label))
    for label in unique_label:
        b += struct.pack('i', len(label))
        for c in label:
            b += struct.pack('c', c)
    for layer in model.layers:
        for w in layer.get_weights():
            for v in w.astype(np.float32).reshape(-1):
                b += struct.pack('f', v)
    with open(filename, 'wb') as fp:
        fp.write(b)


if __name__ == '__main__':

    nb_dim = 48
    batch_size = 128
    nb_epoch = 200

    print 'Load data'
    X, y, unique_label = load_data()
    n = X.shape[0]

    print 'Split data into train set and validation set'
    idx = np.random.permutation(n)
    n_train = int(n * 0.9) / batch_size * batch_size
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_valid, y_valid = X[idx[n_train:]], y[idx[n_train:]]
    steps_per_epoch = 10 * n_train / batch_size

    print 'Build model'
    model = build_model(nb_dim, len(unique_label))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

    print 'Fit'
    earlystopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath='weights.hdf5',
                                verbose=0, save_best_only=True)

    datagen = ImageDataGenerator(
        zoom_range=[0.9, 1.05],
        rotation_range=20)

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystopping, checkpointer],
                        epochs=nb_epoch, verbose=1)

    model.load_weights('weights.hdf5')
    print 'Testing on keras:', (model.predict_classes(X) == y.argmax(axis=1)).mean()

    print 'Dump results to binary'
    dump_weights('cnn-result.bin', model, unique_label)
