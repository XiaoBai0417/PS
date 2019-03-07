
import cv2
import numpy as np
from utils import*

image = get_image("IMG_0237.JPG")
mask = np.zeros(image.shape[:2], dtype = np.uint8)

orb = cv2.ORB_create(10000)
kp1, des1 = orb.detectAndCompute(image,None)
points = cv2.KeyPoint_convert(kp1)
hull = cv2.convexHull(points)

hull = hull.astype(int)
hull = hull.ravel()
hull = hull.reshape(hull.shape[0]/2,2)

cv2.polylines(mask, hull, True, (0, 255, 0), 2)
cv2.fillPoly(mask, hull, 255)

mask = image
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask to Image", masked)


