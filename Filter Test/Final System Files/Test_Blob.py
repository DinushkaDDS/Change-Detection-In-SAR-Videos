from Filter_Frame import *
import cv2
import numpy as np

# Setup SimpleBlobDetector parameters. More details in https://www.learnopencv.com/blob-detection-using-opencv-python-c/
params = cv2.SimpleBlobDetector_Params()
cap = cv2.VideoCapture('/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/CleanNew.mp4')

# Change thresholds for intensity. hence the blobs are dark thresholds should be low
params.minThreshold = 0
params.maxThreshold = 125

# Filter by Area. Need to find a better threshold calculating mechanism
params.filterByArea = True
params.minArea = 100  #Should be set according to a
                        # ratio between considering area and actual expecting object size
params.maxArea = np.pi*12*12


# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 0.9

# Filter by Inertia  // lower values are better because when moving Inertia tends to become 0
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 0.9


# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # frame = filterFrame(frame)

    frame = cv2.equalizeHist(frame)

    keypoints = detector.detect(frame)



    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    for i in keypoints:
        print(i.response)

    cv2.imshow('frame', im_with_keypoints)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue
    # time.sleep(0.5)

else:
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
print("Successfully Completed!")