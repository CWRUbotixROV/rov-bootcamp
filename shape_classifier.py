import cv2

def otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverting image (switching black and white)
    # thresh = (255-thresh)

    return thresh


def get_contours(image, thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions to calculate max_area
    height, width, channels = image.shape
    max_area = height * width * .8

    # Draw rectangle around central contour
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if 1000 < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
