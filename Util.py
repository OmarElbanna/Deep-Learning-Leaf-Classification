import matplotlib.pyplot as plt
import cv2
from torch import nn

def plot_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def preprocess_img(img, max_width, max_height):
    img_width = img.shape[1]
    img_height = img.shape[0]
    if max_width-img_width < max_height-img_height:
        new_width = max_width
        aspect_ratio = img_height / img_width
        new_height = min(int(aspect_ratio * new_width), max_height)

        top = int((max_height - new_height) / 2)
        bottom = max(max_height - new_height - top, 0)
        left = 0
        right = 0
    else:
        new_height = max_height
        aspect_ratio = img_width / img_height
        new_width = min(int(aspect_ratio * new_height), max_width)

        top = 0
        bottom = 0
        left = int((max_width - new_width) / 2)
        right = max(max_width - new_width - left, 0)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=0)
    return img

def calculate_pool_dim(dim, kernel_size, stride):
    return int((dim - kernel_size) / stride + 1)

def calculate_conv_dim(dim, kernel_size, stride, padding):
    return int((dim - kernel_size + 2*padding) / stride + 1)

def apply_conv(img_width, img_height, in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    img_width = calculate_conv_dim(img_width, kernel_size, stride, padding)
    img_height = calculate_conv_dim(img_height, kernel_size, stride, padding)
    return conv, img_width, img_height

def apply_pool(img_width, img_height, kernel_size, stride):
    pool = nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride
    )
    img_width = calculate_pool_dim(img_width, kernel_size, stride)
    img_height = calculate_pool_dim(img_height, kernel_size, stride)
    return pool, img_width, img_height