import cv2
import shape_classifier
import utils

images, shape_label = utils.get_all_training_data()

image = cv2.imread(images[100])

otsu_image = shape_classifier.otsu(image)
shape_classifier.get_contours(image, otsu_image)

cv2.imshow("image", image)
cv2.waitKey(0)

