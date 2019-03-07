# encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import *

img22 = get_image("IMG_0237.JPG")
print img22.shape
img1,_ = get_image('IMG_0824_1.JPG', True)
img2,_ = get_image('IMG_0824_1.JPG', True)


#img1 = Gaussian_Blur(img1)
#img2 = median(img2)
#img2 = Gaussian_Blur(img2)

orb = cv2.ORB_create(10000)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
points = cv2.KeyPoint_convert(kp2)

hull = cv2.convexHull(points)
print hull

index = points.argsort(axis=0)

#right one point
right_o_x = int(points[index[len(points)-1,0],0])
right_o_y = int(points[index[len(points)-1,0],1])

# left one points
left_o_x = int(points[index[0,0],0])
left_o_y = int(points[index[0,0],1])

# top one points
top_o_x = int(points[index[0,1],0])
top_o_y = int(points[index[0,1],1])

bottom_o_x = int(points[index[len(points)-1,1],0])
bottom_o_y = int(points[index[len(points)-1,1],1])

cnt = np.array([[right_o_x, right_o_y], [top_o_x, top_o_y],
                [left_o_x, left_o_y], [bottom_o_x, bottom_o_y]])

w = right_o_x - left_o_x
h = bottom_o_y - top_o_y

cv2.rectangle(img2, (left_o_x, top_o_y), (left_o_x + w, top_o_y + h), (0, 255, 0), 2)
cv2.imshow('show', img2)

bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1=img1,keypoints1=kp1,img2=img2,keypoints2=kp2,
                    matches1to2=matches, outImg=img2, flags=2)
plt.imshow(img3),plt.show()

cv2.waitKey(0)
