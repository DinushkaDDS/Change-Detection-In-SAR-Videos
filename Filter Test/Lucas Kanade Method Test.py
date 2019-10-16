import numpy as np
import cv2
from numpy.linalg import inv


def optical_flow(windowCurrent, windowPast):
    # Optical flow corresponding to the window 1

    Ix_sobel = cv2.Sobel(windowCurrent, cv2.CV_64F, 1, 0, ksize=-1)
    abs_Ix = np.absolute(Ix_sobel)
    Ix = np.uint8(abs_Ix)

    Iy_sobel = cv2.Sobel(windowCurrent, cv2.CV_64F, 0, 1, ksize=-1)
    abs_Iy = np.absolute(Iy_sobel)
    Iy = np.uint8(abs_Iy)

    col1 = Ix.flatten('F')
    col2 = Iy.flatten('F')

    A = np.concatenate((col1[np.newaxis, :], col2[np.newaxis, :])).transpose()

    b = windowCurrent - windowPast
    b = b.flatten().transpose()
    try:
        v = np.matmul(np.matmul(inv(np.matmul(A.T, A)), A.T), b)
    except:
        return (0,0)
    else:
        return v

def optical_flow_imageWindows(imageCurrent, imagePast, window_size=1, steps = 25):

    outList = []

    image1_shape = imageCurrent.shape
    window_length = 2*window_size +1

    window1 = np.zeros((window_length, window_length), np.int16)
    window2 = np.zeros((window_length, window_length), np.int16)

    for h in range(window_size , image1_shape[0] - window_size, steps):
        for w in range(window_size , image1_shape[1] - window_size, steps):
            window_W = 0
            for i in range(h - window_size, h + window_size + 1):
                window_H = 0
                for j in range(w - window_size, w + window_size + 1):
                    window1[window_W][window_H] = imageCurrent[i][j]
                    window2[window_W][window_H] = imagePast[i][j]
                    window_H = window_H + 1
                else:
                    window_W = window_W + 1
            else:
                flow = optical_flow (window1, window2)
                if(flow[0] == 0 and flow[1] == 0):
                    continue
                else:
                    outList.append(((w, h), flow))
    return outList



imagePast = cv2.imread('Screenshot (7).png')
imageCurrent = cv2.imread('Screenshot (8).png')

im1Gray = cv2.cvtColor(imageCurrent, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(imagePast, cv2.COLOR_BGR2GRAY)

lines = optical_flow_imageWindows(im1Gray, im2Gray)

for line in lines:

    im1Gray =  cv2.line(im1Gray, line[0],(int(round(line[0][0]+line[1][0])),
                                          int(round(line[0][1] + line[1][1]))), 255, 1)

    im2Gray = cv2.line(im2Gray, line[0], (int(round(line[0][0] + line[1][0])),
                                          int(round(line[0][1] + line[1][1]))), 255, 1)

cv2.imshow('test1', im1Gray)
cv2.imshow('test2', im2Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
