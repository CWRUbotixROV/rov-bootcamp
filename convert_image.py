import os

import cv2
import numpy as np

AREA_WEIGHT = 100
SMOOTHNESS_WEIGHT = 150
BRIGHTNESS_WEIGHT = 600
EDGE_WEIGHT = 25
BIAS = 100


def score_contour(img, area, perimeter, box, mean_hsv):
    area_score = np.sqrt(area) * AREA_WEIGHT / 200
    smoothness_score = area / perimeter ** 2 * 10 * SMOOTHNESS_WEIGHT
    brightness_score = -mean_hsv[2] / 255 * BRIGHTNESS_WEIGHT

    edge_score = 0
    corners = ((box[0], box[1]), (box[0], box[1] + box[3]),
               (box[0] + box[2], box[1]), (box[0] + box[2], box[1] + box[3]))
    for corner in corners:
        if corner[0] == 0 or corner[0] == img.shape[1]:
            edge_score -= EDGE_WEIGHT
        if corner[1] == 0 or corner[1] == img.shape[0]:
            edge_score -= EDGE_WEIGHT

    total_score = area_score + smoothness_score + brightness_score + edge_score

    return total_score + BIAS


def convert_image(img):
    """
    Automatically find the shape in the image and crop to it

    :param img: The input image
    :return: The processed image
    """

    img = cv2.resize(img, (640, 480))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    ret, blob = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blob = cv2.bitwise_not(blob)

    contours, hierarchy = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cropped_blobs = []
    scores = []
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        box = cv2.boundingRect(contour)

        if box[2] == img.shape[1] or box[3] == img.shape[0]:
            continue

        if area < 1000:
            continue

        mask = np.zeros(blob.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(img_hsv, mask=mask)

        score = score_contour(img, area, perimeter, box, mean_color)
        if score < 0:
            continue

        cropped = img_gray[box[1]:box[1] + box[3],
                           box[0]:box[0] + box[2]]

        resized = cv2.resize(cropped, (100, 100))

        _, cropped_blob = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cropped_blob = cv2.bitwise_not(cropped_blob)

        cropped_blobs.append(cropped_blob)
        scores.append(score)
        boxes.append(box)

    return cropped_blobs, scores, boxes


if __name__ == "__main__":
    if not os.path.exists(f"processed/None"):
        os.makedirs(f"processed/None")
    for subdir in os.listdir("training"):
        if not os.path.exists(f"processed/{subdir}"):
            os.makedirs(f"processed/{subdir}")

        for file in os.listdir(f"training/{subdir}"):
            cropped_imgs, scores, _ = convert_image(cv2.imread(f"training/{subdir}/{file}"))
            if len(cropped_imgs) == 0:
                continue

            filename = file.split('.')[0]
            extension = file.split('.')[1]

            best_score = np.max(scores)
            best_i = scores.index(best_score)
            for i, cropped_image in enumerate(cropped_imgs):
                if i == best_i:
                    cv2.imwrite(f"processed/{subdir}/{file}", cropped_image)
                    pass
                else:
                    cv2.imwrite(f"processed/None/{filename}_{i}.{extension}", cropped_image)

            # cv2.imshow("result", cropped_img)
            # cv2.waitKey()
