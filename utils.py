
import cv2
import numpy as np

def get_image(path, resize = False):
    img = cv2.imread(path)

    #img = equalizeHist(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    if resize:
        size = (int(width * 0.1), int(height * 0.1))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return img, gray

def median(image):
    dst = cv2.medianBlur(image, 9)
    return dst

def Gaussian_Blur(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return blurred

def equalizeHist(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

def Thresh_and_blur(gradient):
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 20, 255, cv2.THRESH_OTSU)
    return thresh

def Sobel_gradient(blurred):

    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

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

def Sobel_gradient(blurred):

    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient

def Thresh_and_blur(gradient):
    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    #(_, thresh) = cv2.threshold(blurred, 20, 255, cv2.THRESH_OTSU)

    return thresh

def image_morphology(thresh):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=10)
    closed = cv2.dilate(closed, None, iterations=10)

    return closed

def findcnts_and_box_point(closed):

    cnts, _= cv2.findContours(closed.copy(),
                                cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box

def drawcnts_and_cut(original_img, box):

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


def Grab(img, dest_name, enhance = False):

    #img = cv2.GaussianBlur(img, (9, 9), 0)
    #img = median(img)


    height, width, _ = img.shape
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, height-10, width-10)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_result = img * mask2[:, :, np.newaxis]
    return img_result
    #cv2.imwrite(dest_name, img_result)