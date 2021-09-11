from settings import SHAPES

from os import walk, makedirs
from os.path import join

import cv2
import imutils

'''Returns two lists. The first list contains all the paths of the training images.
The second list is a list of the shape strings each image is classified as'''
def get_all_training_data():
    paths = []
    shape_label = []
    for shape in SHAPES:
        folder = join('training', shape)
        (_, _, filenames) = next(walk(folder))
        for filename in filenames:
            paths.append(join(folder, filename))
            shape_label.append(shape)
    return paths, shape_label

'''Pops up a window to browse through images put through a function.
Use the 'a' and 'd' keys to navigate through the images. Use 'esc' to clsos.
filepaths - List of file path strings to open
func - A function that takes one image as input and output an image'''
def browse_images(filepaths, func):
    idx = 0
    img = func(cv2.imread(filepaths[idx]))
    cv2.imshow('Image', imutils.resize(img, width=500))

    while True:
        k = cv2.waitKey(0) & 0xFF
        
        if k == 97 and idx > 0:
            idx -= 1
            img = func(cv2.imread(filepaths[idx]))
            cv2.imshow('Image', imutils.resize(img, width=500))
        elif k == 100 and idx < len(filepaths) - 1:
            idx += 1
            img = func(cv2.imread(filepaths[idx]))
            cv2.imshow('Image', imutils.resize(img, width=500))
        elif k == 27:
            cv2.destroyAllWindows()
            break