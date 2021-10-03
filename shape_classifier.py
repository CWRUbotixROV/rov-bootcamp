import cv2
import numpy as np

def otsu(image):
    """
    Performs otsu thresholding on input image
    :param image: original image
    :return: image after otsu thresholding
    """

    image = cv2.resize(image, (640, 480))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)

    return gray, thresh

def get_contours(gray, thresh):
    """
    Finds the best contour in an image and returns it
    :param gray: image in gray scale from otsu
    :param thresh: thresh from otsu
    :return: best contour from image
    """

    height, width = gray.shape

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_diff = 0
    best_contour = None

    # Iterate through each contour and find the best one
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out image border
        if h == height or w == width:
            continue

        # Filter out small rectangles
        if cv2.contourArea(cnt) < 1000:
            continue

        kernel_size = max(w // 10, h // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        contour_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(contour_mask, [cnt], 0, 255, -1)

        border_mask = cv2.dilate(contour_mask, kernel)
        contour_mask_inv = cv2.bitwise_not(contour_mask)
        border_mask = cv2.bitwise_and(border_mask, contour_mask_inv)

        inside_mean, inside_std = mean_std(gray, contour_mask)
        border_mean, border_std = mean_std(gray, border_mask)

        diff = ((border_mean - inside_mean) / inside_std) + ((border_mean - inside_mean) / border_std)

        # If ... set current contour to best_contour
        if diff > max_diff:
            max_diff = diff
            best_contour = cnt

    # Cropped gray image from the bounding rectangle of the best contour and resize
    x, y, w, h = cv2.boundingRect(best_contour)

    cropped = gray[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, (100, 100))

    # Draws rectangle on to gray image
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cropped

def mean_std(img, mask):
    """

    :param img:
    :param mask:
    :return:
    """

    mean, std = cv2.meanStdDev(img, mask=mask)
    return mean[0][0], std[0][0]

def convert_image(image):
    gray, thresh = otsu(image)
    cropped = get_contours(gray, thresh)

    return cropped

def draw_contours(image):
    """
    Draws a box around the best contour in the image
    :param image: image with shape
    :return: image with rectangle around shape
    """
    gray, thresh = otsu(image)
    get_contours(gray, thresh)

    return gray



