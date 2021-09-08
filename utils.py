from settings import SHAPES

from os import walk, makedirs
from os.path import join

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