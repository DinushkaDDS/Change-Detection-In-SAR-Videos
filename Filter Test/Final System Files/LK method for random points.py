import numpy as np
import cv2

fileName = "/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/CleanNew.mp4"

videoFile = cv2.VideoCapture(fileName)


#Compare the points and points next and return a array with deviated vlaues(Need to change the
#       threshold method like taking the average of the all the optical flow magnitudes)
def filterPoints(points, points_next, threshold=2):

    interest_points = []
    for i in range(len(points)):
        x_diff = points_next[i][0] - points[i][0]
        y_diff = points_next[i][1] - points[i][1]

        if( np.sqrt(x_diff**2 + y_diff**2) >= threshold):
            interest_points.append([points_next[i], points[i]])
    print(len(interest_points))
    return interest_points


#Threshold Calculation method need to implement if required
def calculateThreshold(array1, array2, x, y, stepSize, windowSize = 1):
    return


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

pointStep = 50

# Capture the first reference frame
ret, frameRef = videoFile.read()
while(ret != True):
    ret, frameRef = videoFile.read()
frameRef_gray = cv2.cvtColor(frameRef, cv2.COLOR_BGR2GRAY)

points =  np.array([[[0,0]]], dtype=np.float32)
for i in range(0, frameRef.shape[0], pointStep):
    for j in range(0, frameRef.shape[1], pointStep):
        points = np.append(points, [[[np.float32(j),np.float32(i)]]], axis = 0)

#numOfPoints = len(points)

# Create a mask image for drawing purposes
mask = np.zeros_like(frameRef)
pointsInterest_curr = np.array([[[0, 0]]], dtype=np.float32)
pointsInterest_count = [0]

# Read until video is completed
while videoFile.isOpened():

    # Capturing the continuing frames
    ret, frame = videoFile.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        # calculate optical flow
        points_nxt, st, err = cv2.calcOpticalFlowPyrLK(frameRef_gray, frame_gray, points,
                                                       None, **lk_params)

        pointsInterest_nxt, st_interest, err_interest = cv2.calcOpticalFlowPyrLK(frameRef_gray,
                                                                frame_gray, pointsInterest_curr,
                                                                                 None, **lk_params)

        # Select good points
        good_new = points_nxt[st == 1]
        good_old = points[st == 1]

        #Implementation of point filtering based on opticalflow magnitude
        #implementatio of point filtering based on whether it is inside a blob like object
        interestPoints = filterPoints(good_old, good_new)

        #Drawing the interesting points in the frame
        pointsTemp = np.array([[[0, 0]]], dtype=np.float32)
        for points in interestPoints:
            new = points[0]
            old = points[1]
            a, b = new.ravel()
            c, d = old.ravel()
            pointsTemp = np.append(pointsTemp, [[[a, b]]], axis=0)
            pointsInterest_count.append(0)
        pointsTemp = np.delete(pointsTemp, 0, axis=0)


        for i in range(len(pointsInterest_curr)):

            if(st_interest[i] == 0):
                pointsInterest_count[i] = pointsInterest_count[i] + 1
            else:

                x_diff = pointsInterest_nxt[i][0][0] - pointsInterest_curr[i][0][0]
                y_diff = pointsInterest_nxt[i][0][1] - pointsInterest_curr[i][0][1]

                # print(points[i] , "----->" , points_next[i])
                if (np.sqrt(x_diff ** 2 + y_diff ** 2) >= 2):
                    new = pointsInterest_nxt[i]
                    old = pointsInterest_curr[i]
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), [0, 255, 0], 1)
                    frame = cv2.circle(frame, (a, b), 2, [255, 0, 0], -1)
                    pointsInterest_count[i] = 0
                else:
                    pointsInterest_count[i] = pointsInterest_count[i] + 1

        temp = len(pointsInterest_count) -1
        while(temp != 0):
            if(pointsInterest_count[temp]>=3):
                pointsInterest_nxt = np.delete(pointsInterest_nxt, temp, axis=0)
                pointsInterest_count.pop(temp)
            temp = temp - 1

        pointsInterest_nxt = np.concatenate((pointsInterest_nxt, pointsTemp), axis=0)

        frame = cv2.add(frame, mask)

        cv2.imshow('Frame', frame)

        #Setting the points to the new calculation round
        frameRef_gray = frame_gray
        points = points_nxt
        pointsInterest_curr = pointsInterest_nxt
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# Cleaning up the memory
videoFile.release()
cv2.destroyAllWindows()

