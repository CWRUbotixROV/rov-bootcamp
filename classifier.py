import cv2
import numpy as np

def threshold(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return grey, th

def choose_contour(grey, thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width, height = grey.shape

    max_diff = 0
    best_contour = None

    cv2.imshow('Grey', grey)
    
    for contour in contours:
        #print('Area: ' + str(cv2.contourArea(contour)))
        if cv2.contourArea(contour) < 0.03 * width * height:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        kernel_size = max(w // 10, h // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        contour_mask = np.zeros(grey.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        border_mask = cv2.dilate(contour_mask, kernel)
        contour_mask_inv = cv2.bitwise_not(contour_mask)
        border_mask = cv2.bitwise_and(border_mask, contour_mask_inv)

        contour_img = cv2.bitwise_and(grey, contour_mask)
        border = cv2.bitwise_and(grey, border_mask)
        #cv2.imshow('Contour', contour_img)
        #cv2.imshow('Border', border)


        inside_mean, inside_std = get_mean_std(grey, contour_mask)
        border_mean, border_std = get_mean_std(grey, border_mask)

        diff = ((border_mean - inside_mean) / inside_std) + ((border_mean - inside_mean) / border_std)

        if diff > max_diff:
            max_diff = diff
            best_contour = contour
    
    x, y, w, h = cv2.boundingRect(best_contour)
    cv2.rectangle(grey,(x,y),(x+w,y+h),0,5)
    return contour

def get_mean_std(img, mask):
    mean, std = cv2.meanStdDev(img, mask=mask)
    return mean[0][0], std[0][0]

def draw_contour(img):
    grey, thresh = threshold(img)
    choose_contour(grey, thresh)
    return grey