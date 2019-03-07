import os
import cv2
import time
import numpy as np

def get_image(path):
    img = cv2.imread(path)
    size = (256, 256)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img

def change_color(input_name):
    img = get_image(input_name)
    img[img!=0] = 255

    print img!=0

    cv2.imshow('original_img', img)
    cv2.waitKey(0)


change_color("IMG_0824.JPG")

