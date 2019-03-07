# encoding: utf-8

import os
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from utils import *

def clip_min_area(input_name, dest_name):

    image, _ = get_image(input_name, False)
    print type(image)
    #img1 = Gaussian_Blur(image)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(image, None)
    points = cv2.KeyPoint_convert(kp1)

    points = np.array(points, dtype=int)
    print points

    rect = cv2.minAreaRect(points)
    box = np.int0(cv2.boxPoints(rect))

    roi_t = np.asarray(box)
    roi_t = np.expand_dims(roi_t, axis=0)
    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)
    mask = im

    masked = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(dest_name, masked)

    #draw_img = cv2.drawContours(img1.copy(), [box], -1, (0, 0, 255), 3)
    return len(points)

def get_convex(input_name,dest_name):

    image,_ = get_image(input_name, False)
    #img1 = Gaussian_Blur(img1)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(image,None)
    points = cv2.KeyPoint_convert(kp1)

    points = np.array(points, dtype=int)

    rect = cv2.minAreaRect(points)
    box = np.int0(cv2.boxPoints(rect))

    roi_t = np.asarray(box)
    roi_t = np.expand_dims(roi_t, axis=0)
    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)
    mask = im

    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(dest_name, masked)

    # draw_img = cv2.drawContours(img1.copy(), [box], -1, (0, 0, 255), 3)
    #hull = cv2.convexHull(points)

    #hull = hull.astype(int)
    #hull = hull.ravel()
    #hull = hull.reshape(hull.shape[0]/2,2)
    #cv2.polylines(img1, [hull], True, (0, 255, 0), 2)

    cv2.imwrite(dest_name, image)

    return len(points)


def convex_all(dir, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        get_convex(sample_name, label_path)


num_of_key = clip_min_area("IMG_9822.JPG", "temp_IMG_9822.JPG")
print num_of_key

#draw_img, _ = clip_min_area("IMG_0824.JPG", None)



cv2.waitKey(0)

#clip_min_area()

