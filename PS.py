# encoding:utf-8
from utils import *
import os

import matplotlib.pyplot as plt


# limit roi box and mask
def get_min_mask_area(image, min_area_box):
    roi_t = np.asarray(min_area_box)
    roi_t = np.expand_dims(roi_t, axis=0)

    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)

    mask = im
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


# cv2.imwrite(dst_name, masked)


def gradient_and_box(gray):
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)

    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)

    box = findcnts_and_box_point(closed)
    return box


def orb_and_box(img1):
    # img1 = Gaussian_Blur(img1)
    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    points = cv2.KeyPoint_convert(kp1)

    index = points.argsort(axis=0)

    # right one point
    right_o_x = int(points[index[len(points) - 1, 0], 0])
    right_o_y = int(points[index[len(points) - 1, 0], 1])

    # left one points
    left_o_x = int(points[index[0, 0], 0])
    left_o_y = int(points[index[0, 0], 1])

    # top one points
    top_o_x = int(points[index[0, 1], 0])
    top_o_y = int(points[index[0, 1], 1])

    bottom_o_x = int(points[index[len(points) - 1, 1], 0])
    bottom_o_y = int(points[index[len(points) - 1, 1], 1])

    cnt = np.array([[right_o_x, right_o_y], [top_o_x, top_o_y],
                    [left_o_x, left_o_y], [bottom_o_x, bottom_o_y]])

    w = right_o_x - left_o_x
    h = bottom_o_y - top_o_y

    # crop_img = img1[top_o_y -10:top_o_y + h+10, left_o_x-10:left_o_x + w +10]
    crop_img = img1[top_o_y:top_o_y + h, left_o_x:left_o_x + w]

    return crop_img


def get_gradient_rect_area(image, gray):
    min_area_box = gradient_and_box(gray)

    Xs = [i[0] for i in min_area_box]
    Ys = [i[1] for i in min_area_box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = image[y1:y1 + hight, x1:x1 + width]

    return crop_img

    # cv2.imwrite(dst_name, crop_img)


def check_important_points(img1, convex):
    for i in convex:
        x, y = i.ravel()
        crop_img = img1[y - 5: y + 5, x - 5: x + 5]

        print x, y

        print img1[x, y]
        cv2.circle(img1, (x, y), 3, 255, 0)


# apply gradient to get min box
def step1(input_name, dst_name):
    image, gray = get_image(input_name, True)

    # roi = get_gradient_rect_area(image, gray)
    # roi = orb_and_box(image)
    # roi =Grab(image,dst_name)

    cv2.imwrite(dst_name, image)


# def get_resize()

def step2(left_name, right_name, result_name):
    img1, _ = get_image(left_name, False)
    img2, _ = get_image(right_name, False)

    orb = cv2.ORB_create(100000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    points = cv2.KeyPoint_convert(kp1)

    index = points.argsort(axis=0)

    # right one point
    right_o_x = int(points[index[len(points) - 1, 0], 0])
    right_o_y = int(points[index[len(points) - 1, 0], 1])

    # left one points
    left_o_x = int(points[index[0, 0], 0])
    left_o_y = int(points[index[0, 0], 1])

    # top one points
    top_o_x = int(points[index[0, 1], 0])
    top_o_y = int(points[index[0, 1], 1])

    bottom_o_x = int(points[index[len(points) - 1, 1], 0])
    bottom_o_y = int(points[index[len(points) - 1, 1], 1])

    cnt = np.array([[right_o_x, right_o_y], [top_o_x, top_o_y],
                    [left_o_x, left_o_y], [bottom_o_x, bottom_o_y]])

    w = right_o_x - left_o_x
    h = bottom_o_y - top_o_y

    crop_img = img2[top_o_y:top_o_y + h, left_o_x:left_o_x + w]

    cv2.imwrite(result_name, crop_img)


def get_step1(dir, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    iter = 0
    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        step1(sample_name, label_path)
        iter = iter + 1
        if iter % 20 == 0:
            print iter


def get_step2(left_dir, right_dir, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    iter = 0
    dirs = os.listdir(left_dir)
    for file in dirs:
        label_path = result_path + "//" + file
        left_name = left_dir + "//" + file

        right_name = right_dir + "//" + file

        step2(left_name, right_name, label_path)
        iter = iter + 1
        if iter % 20 == 0:
            print iter


def step3(input_name, dst_name):
    img1, _ = get_image(input_name, False)
    # img1 = Gaussian_Blur(img1)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(img1, None)

    points = cv2.KeyPoint_convert(kp1)
    # ch2d=spatial.ConvexHull(points)
    hull = cv2.convexHull(points)

    hull = hull.astype(int)
    # hull = hull.ravel()
    # hull = hull.reshape(hull.shape[0] / 2, 2)

    roi_t = np.asarray(hull)
    roi_t = np.expand_dims(roi_t, axis=0)

    im = np.zeros(img1.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)

    mask = im
    masked = cv2.bitwise_and(img1, img1, mask=mask)
    cv2.imwrite(dst_name, masked)
    # return masked

    # cv2.polylines(img1, [hull], True, (0, 255, 0), 2)


# cv2.imwrite(dst_name, img1)

def step4(input_name, dst_name):
    img1, _ = get_image(input_name, False)

    # img1 = Gaussian_Blur(img1)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(img1, None)

    points = cv2.KeyPoint_convert(kp1)
    # ch2d=spatial.ConvexHull(points)
    hull = cv2.convexHull(points)

    hull = hull.astype(int)
    hull = hull.ravel()
    hull = hull.reshape(hull.shape[0] / 2, 2)

    # cv2.polylines(img1, [hull], True, (0, 255, 0), 2)

    for i in hull:
        x, y = i.ravel()
        print x, y
        print img1[x, y]
        cv2.circle(img1, (x, y), 3, 255, 0)

    plt.imshow(img1), plt.show()


def get_step3(dir, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    iter = 0
    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        step3(sample_name, label_path)
        iter = iter + 1
        if iter % 20 == 0:
            print iter


def step5(input_name, dst_name):
    img1, _ = get_image(input_name, False)
    h, w, _ = img1.shape

    for i in range(h):
        for j in range(w):
            b = img1[i, j, 0]
            g = img1[i, j, 1]
            r = img1[i, j, 2]

            mean = (b + g + r) / 3
            min_temp = min(b, g, r)
            max_temp = max(b, g, r)

            if max_temp == 0:
                continue
            if max_temp - min_temp > 20:
                continue
            if min_temp < 10:
                continue
            if max_temp > 200:
                continue

            if float(min_temp) / max_temp > 0.7:
                img1[i, j] = [0, 0, 0]

                # print max_

    cv2.imwrite(dst_name, img1)

    # cv2.circle(img1, (x, y), 3, 255, 0)

    # plt.imshow(img1), plt.show()


def get_step5(dir, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    iter = 0
    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        step5(sample_name, label_path)
        iter = iter + 1
        if iter % 20 == 0:
            print iter


# 先grab_cut
# 原始图像去掉阴影
# 寻找外接多边形
# 多边形裁剪grab-cut
def step6(grab_name, init_name, dst_name):
    img1, _ = get_image(init_name, False)
    img2, _ = get_image(grab_name, False)

    # img1 = Gaussian_Blur(img1)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(img1, None)

    points = cv2.KeyPoint_convert(kp1)
    # ch2d=spatial.ConvexHull(points)
    hull = cv2.convexHull(points)

    hull = hull.astype(int)
    hull = hull.ravel()
    hull = hull.reshape(hull.shape[0] / 2, 2)

    mask = get_min_mask_area(img2, hull)
    cv2.imwrite(dst_name, mask)


def get_step6(grab_name, init_name, dst_name):
    if not os.path.exists(dst_name):
        os.makedirs(dst_name)
    iter = 0
    dirs = os.listdir(init_name)
    for file in dirs:

        label_path = dst_name + "//" + file
        left_name = grab_name + "//" + file
        right_name = init_name + "//" + file

        step6(left_name, right_name, label_path)
        iter = iter + 1
        if iter % 20 == 0:
            print iter


# step1("IMG_0240.JPG", "IMG_0240_.JPG")
# step5("IMG_0782.JPG", "IMG_0782_.JPG")
get_step5("1-40_resize", "1-40_sub__")
# get_step1("1-40_glabcut_convex", "1-40_glabcut_convex_gradient")
# get_step1("1-40", "1-40_resize")
# get_step3("1-40_glabcut", "1-40_glabcut_convex")
# get_step3("1-40_glabcut_convex", "1-40_glabcut_convex_2")
#get_step6("1-40_glabcut", "1-40_sub__", "result")
