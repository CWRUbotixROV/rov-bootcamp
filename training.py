import cv2
from skimage.feature import hog
from sklearn.svm import SVC
import pickle

import shape_classifier
import utils

def prepare_data():
    """
    Goes through all the images to get the best contour for each one
    :returns: list of cropped images and a list of the corresponding shape labels
    """

    all_images, all_labels = utils.get_all_training_data()

    selected_images = []
    selected_labels = []

    print("Preparing images...")

    for i in range(len(all_images)):
        image = cv2.imread(all_images[i])

        cropped = shape_classifier.convert_image(image)

        if cropped is not None:
            selected_images.append(cropped)
            selected_labels.append(all_labels[i])

    print("Finished preparing images")

    return selected_images, selected_labels

def do_hog():
    """

    """

    images, labels = prepare_data()

    hogs = []

    for image in images:
        hogs.append(hog(image))

    file = open("training_features.obj", "wb");
    pickle.dump((hogs, labels), file)

    svc = SVC()
    svc.fit(hogs, labels)

