import matplotlib.pyplot as plt
import cv2

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