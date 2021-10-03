import unittest
import cv2

import utils
import shape_classifier
import training

class Test(unittest.TestCase):

    def setUp(self):
        self.all_images, self.all_labels = utils.get_all_training_data()

    def test_draw_contours(self):
        utils.browse_images(self.all_images, shape_classifier.draw_contours)

    def test_prepare_data(self):
        training.prepare_data()

    def test_live_video(self):
        video = cv2.VideoCapture(0)

        while True:
            ret, frame = video.read()

            # Space bar to quit video
            key = cv2.waitKey(1)
            if key == 32:
                break

            cv2.imshow("Video", frame)

        cv2.destroyAllWindows()