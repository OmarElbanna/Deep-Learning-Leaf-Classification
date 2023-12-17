import pandas as pd
import os
import matplotlib.pyplot as plt

from Util import *

dataset_train_labels_path = 'dataset/train.csv'
dataset_train_images_path = 'dataset/images'

df = pd.read_csv(dataset_train_labels_path)
labels_list = df['species'].to_numpy()
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