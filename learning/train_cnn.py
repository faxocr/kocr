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

import os
import cv2 as cv


def load_data(input_dirs, nb_dim, pad=3):
    unique_labels = set()
    for input_dir in input_dirs:
        for name in os.listdir(input_dir):
            if name.endswith('.png'):
                unique_labels.add(name[0])
    unique_labels = sorted(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    X, y = [], []
    for input_dir in input_dirs:
        for name in os.listdir(input_dir):
            if not name.endswith('.png'):
                continue
            img = cv.imread(input_dir + name, 0)

            # Thresholding
            img = cv.threshold(img, 255 * 0.7, 255, cv.THRESH_BINARY_INV)[1]
            # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

            # Cropping
            xs, ys = np.where(img > 0)
            img = img[xs.min(): xs.max() + 1, ys.min(): ys.max() + 1]

            # Resizing
            ratio = (nb_dim - 2 * pad) / float(max(img.shape))
            method = cv.INTER_AREA if ratio < 1 else cv.INTER_LINEAR
            img = cv.resize(img, None, fx=ratio, fy=ratio, interpolation=method)

            # Padding
            top = (nb_dim - img.shape[0]) // 2
            left = (nb_dim - img.shape[1]) // 2
            img = cv.copyMakeBorder(img, top, nb_dim - img.shape[0] - top,
                left, nb_dim - img.shape[1] - left, cv.BORDER_CONSTANT, value=0)

            X.append(img)
            y.append(label_to_idx[name[0]])

    X = np.asarray(X, dtype=np.float32) / 255.
    X = X[:, None, ...]
    y = y = np_utils.to_categorical(np.asarray(y, dtype=int))
    unique_labels = np.asarray(unique_labels)

    return X, y, unique_labels


def build_model(nb_dim, nb_output):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', padding='valid', input_shape=(1, nb_dim, nb_dim)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
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


# https://github.com/yu4u/cutout-random-erasing
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_c, img_h, img_w = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        # if pixel_level:
        #     c = np.random.uniform(v_l, v_h, (h, w, img_c))
        # else:
        #     c = np.random.uniform(v_l, v_h)

        c = 0
        input_img[:, top:top + h, left:left + w] = c

        return input_img

    return eraser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_dirs', type=str, nargs='*',
                        default=['raw_num_data/'], help='directories of training images')
    parser.add_argument('--test_dirs', type=str, nargs='*',
                        default=['mustread/'], help='directories of testing images')
    parser.add_argument('--dump_prefix', type=str, default='')
    parser.add_argument('--nb_dim', type=int, default=48,
                        help='dim of images')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epoch', type=int, default=200)
    args = parser.parse_args()

    print ('Load data')
    X, y, unique_label = load_data(args.train_dirs, args.nb_dim)
    n = X.shape[0]

    print ('Split data into train set and validation set')
    idx = np.random.permutation(n)
    n_train = int(int(n * 0.9) / args.batch_size) * args.batch_size
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_valid, y_valid = X[idx[n_train:]], y[idx[n_train:]]
    steps_per_epoch = n_train / args.batch_size

    print ('Build model')
    model = build_model(args.nb_dim, len(unique_label))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                metrics=['accuracy'])

    print ('Fit')
    earlystopping = EarlyStopping(monitor='val_loss', patience=1000)
    checkpointer = ModelCheckpoint(filepath=args.dump_prefix + 'weights.hdf5',
                                   verbose=0, save_best_only=True)

    datagen = ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1)
    # datagen = ImageDataGenerator(preprocessing_function=get_random_eraser())

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystopping, checkpointer],
                        epochs=args.nb_epoch, verbose=1)
    model.load_weights(args.dump_prefix + 'weights.hdf5')

    print ('Dump results to binary')
    dump_weights(args.dump_prefix + 'cnn-result.bin', model, unique_label)

    print ('Testing on validation set:', (model.predict_classes(X_valid) == y_valid.argmax(axis=1)).mean())
    for test_dir in args.test_dirs:
        X_test, y_test, unique_label_test = load_data([test_dir], args.nb_dim)
        y_test = unique_label_test[y_test.argmax(axis=1)]
        y_pred = unique_label[model.predict_classes(X_test)]
        print ('Testing on {}: {}'.format(test_dir, (y_test == y_pred).mean()))

