import cv2
import numpy as np
import os
import tensorflow_datasets as tfds

# EMNIST labels
letter_table = {0: '_', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
    10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's',
    20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '_'}

balanced_table = {0: '0', 1: '1' , 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't', 47: '_'}

digits_table = {0: '0', 1: '1' , 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Pattern for creating images
emnist_images = [
    {'name': 'emnist_number', 'load': 'emnist/digits', 'split': 'train', 'letter': 'none', 'size': 60000},
    {'name': 'emnist_number', 'load': 'emnist/digits', 'split': 'test', 'letter': 'none', 'size': 3000},
    {'name': 'emnist_alphabet_number', 'load': 'emnist/balanced', 'split': 'train', 'letter': 'none', 'size': 60000},
    {'name': 'emnist_alphabet_number', 'load': 'emnist/balanced', 'split': 'test', 'letter': 'none', 'size': 3000},
    {'name': 'emnist_alphabet_lowercase', 'load': 'emnist/letters', 'split': 'train', 'letter': 'lower', 'size': 60000},
    {'name': 'emnist_alphabet_lowercase', 'load': 'emnist/letters', 'split': 'test', 'letter': 'lower', 'size': 3000},
    {'name': 'emnist_alphabet_uppercase', 'load': 'emnist/letters', 'split': 'train', 'letter': 'upper', 'size': 60000},
    {'name': 'emnist_alphabet_uppercase', 'load': 'emnist/letters', 'split': 'test', 'letter': 'upper', 'size': 3000},
]

for emnist in emnist_images:
    # Make directory
    emnist_dir = '../images/{}_{}'.format(emnist['name'], emnist['split'])
    os.makedirs(emnist_dir, exist_ok=True)

    # Load EMNIST dataset
    image, label = tfds.as_numpy(tfds.load(
        emnist['load'],
        split=emnist['split'], 
        batch_size=-1, 
        as_supervised=True,
    ))

    # Resize dataset
    if emnist['size'] > 0:
        image = image[:emnist['size']]

    # Create iamges
    for i, img in enumerate(image):
        img = np.rot90(img, 1)
        img = np.flipud(img)
        img = 255 - img

        letter = "_"
        if emnist['load'] == 'emnist/balanced':
            letter = balanced_table[label[i]]
        elif emnist['load'] == 'emnist/letters':
            letter = letter_table[label[i]]
        elif emnist['load'] == 'emnist/digits':
            letter = digits_table[label[i]]

        if emnist['letter'] == 'upper':
            letter = letter.upper()
        elif emnist['letter'] == 'lower':
            letter = letter.lower()

        filename = '{}/{}_{:0>8}.png'.format(emnist_dir, letter, i)
        print(filename)
        cv2.imwrite(filename, img)
