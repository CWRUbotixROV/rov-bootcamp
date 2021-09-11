import cv2
import color_mask

image = cv2.imread("training/Square/IMG_1161.JPG")

final_image = color_mask.get_mask(image)

cv2.imshow("image", final_image)

cv2.waitKey(0)

