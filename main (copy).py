# encoding:utf-8

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

iters = 0

def get_image(path,resize = False):  # 获取图片
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    # 缩小图像
    if resize:
        size = (int(width * 0.1), int(height * 0.1))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return img, gray

def kmean(img_path):
    original_img, gray = get_image(img_path, False)
    Z = original_img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    #define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
    #Now convert back into uint8, and make original
    imagecenter = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((original_img.shape))
    cv2.imshow('res2', res2)
    cv2.waitKey(0)

def Gaussian_Blur(gray):  # 高斯去噪(去除图像中的噪点)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return blurred

def equalize(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)  # 合并每一个通道 result = cv2.merge((bH, gH, rH))

    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avgB = np.average(nimg[0])
    avgG = np.average(nimg[1])
    avgR = np.average(nimg[2])

    avg = (avgB + avgG + avgR) / 3

    nimg[0] = np.minimum(nimg[0] * (avg / avgB), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avgG), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avgR), 255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def gamma(img, gamma_value = 1.5):
    fi = img / 255.0
    # 伽马变换
    gamma = gamma_value
    out = np.power(fi, gamma)
    out = out * 255
    out = out.astype(np.uint8)
    return out

def median(image):
    dst = cv2.medianBlur(image, 5)
    return dst

def corner(input_name):
    img, gray = get_image(input_name, False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 找到Harris角点
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst) #找到重心
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # #定义迭代次数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria) #返回角点 #绘制
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    cv2.imwrite('subpixel5.png',img)

def Sobel_gradient(blurred):

    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient


def Thresh_and_blur(gradient):  # 设定阈值
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 20, 255, cv2.THRESH_OTSU)

    return thresh

def image_morphology(thresh):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=0)
    closed = cv2.erode(closed, None, iterations=0)

    return closed

def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    cnts, _= cv2.findContours(closed.copy(),
                                cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
    print len(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box

def drawcnts_and_cut(original_img, box):  # 目标图像裁剪
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    return draw_img, crop_img

def extract(input_name, des_name, resize = False):
    img_path = input_name
    save_path = des_name
    original_img, gray = get_image(img_path, resize)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)

    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed)
    draw_img, crop_img = drawcnts_and_cut(original_img, box)

    # 暴力一点，把它们都显示出来看看
    #cv2.imshow('original_img', original_img)
    #cv2.imshow('GaussianBlur', blurred)
    #cv2.imshow('gradX', gradX)
    #cv2.imshow('gradY', gradY)
    #cv2.imshow('final', gradient)
    #cv2.imshow('thresh', thresh)
   # cv2.imshow('closed', closed)
    #cv2.imshow('draw_img', draw_img)
    #cv2.imshow('crop_img', crop_img)
    cv2.imwrite(save_path, closed)
    cv2.waitKey(0)

def Grab1(input_name, dest_name, enhance = False):
    img ,gray = get_image(input_name, True)
    #img = cv2.GaussianBlur(img, (9, 9), 0)

    if enhance:
        img = median(img)

    height, width, _ = img.shape
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, height-10, width-10)  # 划定区域

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # 函数返回值为mask, bgdModel, fgdModel
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 0和2做背景
    img_result = img * mask2[:, :, np.newaxis]  # 使用蒙板来获取前景区域

    cv2.imwrite(dest_name, img_result)
    #return img, img_result,

def Grab2(input_name, dest_name, enhance = False):
    img ,gray = get_image(input_name)

    (b, g, r) = cv2.split(img)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    img = cv2.merge((gradient, gradient, gradient))

    if enhance:
        img = grey_world(img)

    height, width, _ = img.shape
    L = min(height, width)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, height-10, width-10)  # 划定区域

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # 函数返回值为mask, bgdModel, fgdModel
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 0和2做背景
    img_result = img * mask2[:, :, np.newaxis]  # 使用蒙板来获取前景区域

    cv2.imwrite(dest_name, img_result)
    #return img, img_result,

def listDir(rootDir, dstDir):
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            result_path = dstDir
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            else:
                det_name = result_path + "//" + filename
                t0 = time.time()
                #extract(pathname, det_name)
                Grab1(pathname, det_name)
                #extract(pathname, det_name,True)
                print time.time() - t0
        else:
            listDir(pathname)

#listDir("mz-mzxwqkl-hz-kbqnkw-62g","dest")
#corner("IMG_0824.JPG")
result = get_image("IMG_0824.JPG")









