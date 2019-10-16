import cv2
import numpy as np


sharpen = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

#emboss filter with additional weight to middle point
custom = np.array([[-1, -1, 0],
                   [-1, 1.5, 1],
                   [0, 1, 1]])

log = np.array([[0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -14, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]])


filters = [sharpen]


def filterFrame(frame, filters = filters):

    frame = cv2.medianBlur(frame, 3)

    for filter in filters:
        frame = cv2.filter2D(frame, -1, filter)  # -1 is to keep the image depth as same as the source image

    return frame


def filterFrameNormal(frame):

    frame = cv2.equalizeHist(frame)
    frame = filterFrame(frame)

    frame = cv2.fastNlMeansDenoising(frame, None, 20)

    return frame


def filterFrameSP(frame):

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_blurred = cv2.GaussianBlur(frame, (11, 11), 10)
    frame = cv2.addWeighted(frame, 1.0 + 3.0, im_blurred, -3.0, 0)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th3


