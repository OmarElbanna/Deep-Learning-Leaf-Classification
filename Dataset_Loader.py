from pandas import read_csv 
from os import path
from matplotlib.pyplot import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import FloatTensor, LongTensor
from math import ceil
from time import time
import numpy as np

from Util import *

def load_leafs_dataset(test_split_size, batch_size):
    # max_width -> 1633
    # max_height -> 1089
    # number of images -> 990
    # number of categories -> 99
    
    round_digits = 3

    dataset_train_labels_path = 'dataset/train.csv'
    dataset_train_images_path = 'dataset/images'

    time1 = time()
    df = read_csv(dataset_train_labels_path)
    time2 = time()

    print(f'Reading CSV: {round(time2-time1, round_digits)} seconds')

    encoder = LabelEncoder()
    labels_list = df['species'].to_numpy()
    labels_list = encoder.fit_transform(labels_list)

    time3 = time()
    print(f'Encoding Labels: {round(time3-time2, round_digits)} seconds')

    images_list = []
    max_width = 0
    max_height = 0
    for id in df['id']:
        img = imread(path.join(dataset_train_images_path, f'{id}.jpg'))
        images_list.append(img)

        img_width = img.shape[1]
        img_height = img.shape[0]
        max_width = max(max_width, img_width)
        max_height = max(max_height, img_height)

    resized_images_list = []
    for i in range(len(images_list)):
        img = preprocess_img(images_list[i], max_width, max_height)
        resized_images_list.append(img)
    resized_images_list = np.array(resized_images_list)

    time4 = time()
    print(f'Resizing Images: {round(time4-time3, round_digits)} seconds')

    images_train, images_test, labels_train, labels_test = train_test_split(
        resized_images_list,
        labels_list,
        test_size=test_split_size,
        random_state=1,
        shuffle=True
    )

    time5 = time()
    print(f'Splitting Data: {round(time5-time4, round_digits)} seconds')

    images_train = FloatTensor(images_train)
    images_train = images_train.reshape((
        images_train.shape[0],
        1,
        images_train.shape[1],
        images_train.shape[2]
    ))
    labels_train = LongTensor(labels_train)

    images_test = FloatTensor(images_test)
    images_test = images_test.reshape((
        images_test.shape[0],
        1,
        images_test.shape[1],
        images_test.shape[2]
    ))
    labels_test = LongTensor(labels_test)

    time6 = time()
    print(f'Converting To Tensor: {round(time6-time5, round_digits)} seconds')

    num_batches = ceil(images_train.size()[0] / batch_size)
    images_train = [
        images_train[batch_size*y:batch_size*(y+1), :, :, :] for y in range(num_batches)
    ]
    labels_train = [
        labels_train[batch_size*y:batch_size*(y+1)] for y in range(num_batches)
    ]

    time7 = time()
    print(f'Dividing To Batches: {round(time7-time6, round_digits)} seconds')

    return images_train, images_test, labels_train, labels_test