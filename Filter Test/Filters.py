import cv2
import numpy as np

print(cv2.__version__)

sharpen = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

emboss = np.array([[-2, -1, 0],        #Interesting
                   [-1, 1, 1],
                   [0, 1, 2]])

# Second derivative of the image/ Edge detection filter Highly affected by noise
outline = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Added some blur effect to remove the effects of noise
gaussianLaplasian = np.array([[0, 0, -1, 0, 0],
                             [0, -1, -2, -1, 0],
                             [-1, -2, 16, -2, -1],
                             [0, -1, -2, -1, 0],
                             [0, 0, -1, 0, 0]])

unsharpMasking = (-1/256)*np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, -476, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])

sobel = np.array([[-1, -2, -1],    # Not Working
                  [0, 0, 0],
                  [1, 2, 1]])

custom = np.array([[-2, -1, 0],
                   [-1, 1.5, 1],
                   [0, 1, 2]])

custom2 = np.array([[0, 1, 2],
                   [-1, 1.5, 1],
                   [-2, -1, 0]])

cap = cv2.VideoCapture('Videos/Clean.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame = cv2.fastNlMeansDenoising(frame, None, 10, 10, 7)   # Very Slow. Improve the Parameters.
    frame = cv2.medianBlur(frame, 3)

    frame = cv2.filter2D(frame, -1, sharpen)     # -1 is to keep the image depth as same as the source image

    frame = cv2.filter2D(frame, -1, custom)

    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # time.sleep(0.5)

else:
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
print("Successfully Completed!")

# Apply a Change detection algorithm to check the accuracy of the system
# Create the Proposal as well.
