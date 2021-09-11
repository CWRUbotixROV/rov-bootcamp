import pickle
import cv2
import numpy as np
from skimage.util import img_as_float

import transformers
from convert_image import convert_image

SHAPE_COLORS = {
    'Square': (0, 0, 255),
    'Squiggle': (255, 0, 0),
    'Star': (0, 255, 255),
    'Triangle': (0, 255, 0),
    'None': (0, 0, 0)
}

with open("model.dat", 'rb') as model_file:
    model = pickle.load(model_file)

with open("scaler.dat", 'rb') as model_file:
    scalify = pickle.load(model_file)

hogify = transformers.HogTransformer(
    pixels_per_cell=(10, 10),
    cells_per_block=(2, 2),
    orientations=9,
    block_norm='L2-Hys'
)

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()

    key = cv2.waitKey(1)
    if key == 27:
        break  # esc to quit

    possible_images, scores, bounding_boxes = convert_image(img)

    for i, test_img in enumerate(possible_images):
        test_hog = hogify.transform(np.array([img_as_float(test_img)]))
        test_prepared = scalify.transform(test_hog)

        prediction = model.predict(test_prepared)[0]

        if prediction == 'None':
            continue

        box = bounding_boxes[i]
        img = cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), SHAPE_COLORS[prediction], 2)
        img = cv2.putText(img, prediction, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, SHAPE_COLORS[prediction], 1)

    cv2.imshow("Live Detection", img)

cv2.destroyAllWindows()
