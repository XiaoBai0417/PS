import os
import cv2
import time
import numpy as np

def get_image(path):
    img = cv2.imread(path)
    #size = (256, 256)
    #img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img

def change_color(input_name, dest_name):
    img = get_image(input_name)
    img[img!=0] = 255

    height, weight, channels = img.shape

    for row in range(height):
        for col in range(weight):
            for c in range(channels):
                if img[row, col, c] == 255:
                    img[row, col, :] = 255

    cv2.imwrite(dest_name, img)


def get_label(dir, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        change_color(sample_name, label_path)
        print sample_name

#change_color("result_all")
get_label("result_all", "label_all")

