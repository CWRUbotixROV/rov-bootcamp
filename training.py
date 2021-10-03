import shape_classifier
import utils
from skimage.feature import hog
from sklearn.svm import SVC
import pickle

def prepared_images():
    """
    Goes through all the photos to get the best contour for each one
    :returns: list of cropped contours and a list of the corresponding shape labels
    """

    all_images, all_labels = utils.get_all_training_data()

    selected_images = []
    selected_labels = []

    for i in range(len(all_images)):
        image = shape_classifier.convert_image(all_images[i])

        if image is not None:
            selected_images.append(image)
            selected_labels.append(all_labels[i])

    return selected_images, selected_labels

def do_hog():
    """

    :param images: cropped contours from convert_images
    :param labels: corresponding shape labels for images
    """

    images, labels = prepared_images()

    hogs = []

    for image in images:
        hogs.append(hog(image))

    file = open("training_features.obj", "wb");
    pickle.dump((hogs, labels), file)

    svc = SVC()
    svc.fit(hogs, labels)

