from utils import *
import os

#limit roi box and mask
def get_min_mask_area(image, min_area_box, dst_name):

    roi_t = np.asarray(min_area_box)
    roi_t = np.expand_dims(roi_t, axis=0)

    im = np.zeros(image.shape[:2], dtype="uint8")
    cv2.polylines(im, roi_t, 1, 255)
    cv2.fillPoly(im, roi_t, 255)

    mask = im
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(dst_name, masked)

#apply gradient to get min box
def step1(input_name ,dst_name):

    image, gray = get_image(input_name, True)
    img_result = Grab(image,dst_name)

    orb = cv2.ORB_create(10000)
    kp1, des1 = orb.detectAndCompute(img_result, None)
    points = cv2.KeyPoint_convert(kp1)

    points = np.array(points, dtype=int)


    rect = cv2.minAreaRect(points)
    box = np.int0(cv2.boxPoints(rect))

    roi_t = np.asarray(box)
    roi_t = np.expand_dims(roi_t, axis=0)
    im = np.zeros(img_result.shape[:2], dtype="uint8")
    cv2.polylines(img_result, roi_t, 1, 255)
    cv2.fillPoly(img_result, roi_t, 255)
    mask = im

    masked = cv2.bitwise_and(img_result, img_result, mask=mask)
    cv2.imwrite(dst_name, masked)

    #rect = cv2.minAreaRect(points)
    #box = np.int0(cv2.boxPoints(rect))

    #get_min_mask_area(original_img, box, dst_name)

    #blurred = Gaussian_Blur(gray)
    #gradX, gradY, gradient = Sobel_gradient(blurred)

    #thresh = Thresh_and_blur(gradient)
    #closed = image_morphology(thresh)

    #get getminArea
    #box = findcnts_and_box_point(closed)
    #get_min_mask_area(original_img, box, dst_name)

def get_step1(dir, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    iter =0
    dirs = os.listdir(dir)
    for file in dirs:
        label_path = result_path + "//" + file
        sample_name = dir + "//" + file
        step1(sample_name, label_path)
        iter =iter +1
        if iter % 40 == 0:
            print iter

#need glabcut?
def step2():
    return

#need orb?
def step3():
    return

#get_step1("1-40_all", "step1")
get_step1("1-40_all", "1-40_all_grab_orb_mask")





