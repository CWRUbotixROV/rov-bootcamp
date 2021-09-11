import cv2
import numpy as np


def get_mask(image):
    """Converts image to hsv and returns a color mask that filters everything except for white"""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([70, 0, 0], dtype=np.uint8)
    upper_white = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    return mask