import cv2
import training

def live_video():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()

        # Space bar to quit video
        key = cv2.waitKey(1)
        if key == 32:
            break

        cv2.imshow("Video", frame)

    cv2.destroyAllWindows()

training.prepared_images()
