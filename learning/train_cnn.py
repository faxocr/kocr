import numpy as np
np.random.seed(1024)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import struct
import argparse
import sys


def load_data(prefixes, nb_dim):
    unique_label = None
    Xs, ys = [], []
    for prefix in prefixes:
        X, labels = np.load(prefix + 'images.npy'), np.load(prefix + 'labels.npy')
        X = X.astype(np.float32).reshape([-1, 1, nb_dim, nb_dim]) / 255
        y = np.zeros(labels.shape)

        if unique_label is None:
            unique_label = sorted(np.unique(labels))
        else:
            assert sorted(np.unique(labels)) == unique_label

        for i, label in enumerate(unique_label):
            y[labels == label] = i

        y = np_utils.to_categorical(y.astype(np.uint8))

        Xs.append(X)
        ys.append(y)

    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    return Xs, ys, unique_label


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

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_prefixes', type=str, nargs='*',
                        default=[''], help='prefixes of training npy files')
    parser.add_argument('--test_prefixes', type=str, nargs='*',
                        default=[], help='prefixes of testing npy files')
    parser.add_argument('--dump_prefix', type=str, default='')
    parser.add_argument('--nb_dim', type=int, default=48,
                        help='dim of images')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epoch', type=int, default=200)
    args = parser.parse_args()

    print 'Load data'
    X, y, unique_label = load_data(args.train_prefixes, args.nb_dim)
    n = X.shape[0]

    print 'Split data into train set and validation set'
    idx = np.random.permutation(n)
    n_train = int(n * 0.9) / args.batch_size * args.batch_size
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_valid, y_valid = X[idx[n_train:]], y[idx[n_train:]]
    steps_per_epoch = n_train / args.batch_size

    print 'Build model'
    model = build_model(args.nb_dim, len(unique_label))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

    print 'Fit'
    earlystopping = EarlyStopping(monitor='val_loss', patience=1000)
    checkpointer = ModelCheckpoint(filepath=args.dump_prefix + 'weights.hdf5',
                                   verbose=0, save_best_only=True)

    datagen = ImageDataGenerator(
        zoom_range=[0.9, 1.05],
        rotation_range=20)

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystopping, checkpointer],
                        epochs=args.nb_epoch, verbose=1)
    model.load_weights(args.dump_prefix + 'weights.hdf5')

    print 'Dump results to binary'
    dump_weights(args.dump_prefix + 'cnn-result.bin', model, unique_label)

    print 'Testing on validation set:', (model.predict_classes(X_valid) == y_valid.argmax(axis=1)).mean()
    for prefix in args.test_prefixes:
        X_test, y_test, unique_label_test = load_data([prefix], args.nb_dim)
        assert unique_label_test == unique_label
        print 'Testing on {}: {}'.format(prefix, (model.predict_classes(X_test) == y_test.argmax(axis=1)).mean())
