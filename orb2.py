# encoding: utf-8

import os
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from utils import *

img1,_ = get_image('IMG_0237.JPG', False)
img1 = Gaussian_Blur(img1)


orb = cv2.ORB_create(10000)
kp1, des1 = orb.detectAndCompute(img1,None)

points = cv2.KeyPoint_convert(kp1)
#ch2d=spatial.ConvexHull(points)
hull = cv2.convexHull(points)
print hull

hull = hull.astype(int)
hull = hull.ravel()
hull = hull.reshape(hull.shape[0]/2,2)

print hull

cv2.polylines(img1, [hull], True, (0, 255, 0), 2)

#cv2.rectangle(img2, (left_o_x, top_o_y), (left_o_x + w, top_o_y + h), (0, 255, 0), 2)
cv2.imshow('show', img1)

#poly = plt.Polygon(points[ch2d.vertices], fill=None, lw=2, color='r', alpha=0.5)
#ax = plt.subplot(aspect='equal')
#plt.plot(points[:, 0], points[:, 1], 'go')
#for i, pos in enumerate(points):
#    plt.text(pos[0], pos[1], str(i), color='blue')

#ax.add_artist(poly)
#plt.show()



#print len(points)

#for i in hull:
 #   x,y = i.ravel()#将多维数组降位一维
  #  cv2.circle(img1,(x,y),3,255,-1)

#plt.imshow(img1),plt.show()
cv2.waitKey(0)
