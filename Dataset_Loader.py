import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch

from Util import *

def load_leafs_dataset(batch_size):
    # max_width -> 1633
    # max_height -> 1089
    # number of categories -> 99

    dataset_train_labels_path = 'dataset/train.csv'
    dataset_train_images_path = 'dataset/images'
    df = pd.read_csv(dataset_train_labels_path)

    encoder = LabelEncoder()
    labels_list = df['species'].to_numpy()
    labels_list = encoder.fit_transform(labels_list)
    labels_list = np.reshape(labels_list, (batch_size, 1, -1))
    labels_list = torch.from_numpy(labels_list).float()

    images_list = []
    max_width = 0
    max_height = 0
    for id in df['id']:
        img = plt.imread(os.path.join(dataset_train_images_path, f'{id}.jpg'))
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
    resized_images_list = np.reshape(resized_images_list, (batch_size, -1, max_height, max_width))
    resized_images_list = torch.from_numpy(resized_images_list).float()

    return resized_images_list, labels_list