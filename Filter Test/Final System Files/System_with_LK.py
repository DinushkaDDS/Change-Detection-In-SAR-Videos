from Filter_Frame import *
import cv2
import numpy as np

fileName = "/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/CleanNew.mp4"

# Parameters for Lucas Kanade optical flow
lkparams = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

pointStep = 50  #Parameter to provide the point distribution throughout the frame

#Generate Initial values for the lk method
def initialValuesLK(initialFrame, pointStep):

    regularPoints = np.array([[[0, 0]]], dtype=np.float32)
    markedPoints = np.array([[[0, 0]]], dtype=np.float32)
    for i in range(0, initialFrame.shape[0], pointStep):
        for j in range(0, initialFrame.shape[1], pointStep):
            regularPoints = np.append(regularPoints, [[[np.float32(j), np.float32(i)]]], axis=0)
    counterArr = [0]
    #A mask image for drawing purposes
    mask = np.zeros_like(initialFrame)
    return regularPoints, markedPoints, counterArr, mask

#Not required for this script
#
# #Calculate tan value and magnitude of change of a point
# def findTanAndMag(index, pointsCurr, pointsRef):
#     if(pointsCurr.ndim == 3 ):
#
#         x_diff = pointsCurr[index][0][0] - pointsRef[index][0][0]
#         y_diff = pointsCurr[index][0][1] - pointsRef[index][0][1]
#     else:
#         x_diff = pointsCurr[index][0] - pointsRef[index][0]
#         y_diff = pointsCurr[index][1] - pointsRef[index][1]
#
#     tanValue = y_diff/x_diff
#     magnitude = np.sqrt(x_diff**2 + y_diff**2)
#     return tanValue, magnitude
#
# #This function ignores the windowSize number of rows and columns from the frame edges
# def checkThreshold(index, pointsCurr, pointsRef, numOfRows, numOfColumns, windowSize = 1, threshold= 0.15):
#
#     row, column = findRowAndCol(index, numOfRows, numOfColumns)
#
#     tanSum = 0
#     magnitudeSum = 0
#
#     for r in range(row - windowSize, row + windowSize + 1):
#         for c in range(column - windowSize, column + windowSize + 1):
#             tempIndex = findIndex(r,c,numOfColumns)
#             tanVal , magnitude = findTanAndMag(tempIndex, pointsCurr, pointsRef)
#             tanSum = tanSum + tanVal
#             magnitudeSum = magnitudeSum + magnitude
#
#     totalPoints = (2*windowSize + 1)**2
#     avgTan = tanSum/totalPoints
#     avgMag = magnitudeSum/totalPoints
#     currTan, currMag = findTanAndMag(index, pointsCurr, pointsRef)
#
#     varTan = abs(avgTan-currTan)/avgTan
#     varMag = abs(avgMag - currMag)/avgMag
#
#     if(varMag>threshold and varTan>threshold):
#         return True
#     else:
#         return False
#
# #Function to get the matrix coordinate of the given index
# def findRowAndCol(index, numofRows, numofColumns):
#     row = 0
#     column = 0
#     while(index != (numofColumns*row + column)):
#
#         column = column + 1
#         if(column>= numofColumns):
#             column = 0
#             row = row + 1
#
#         if(row>= numofRows):
#             return None
#
#     return row, column
#
# #Function to get the index when row,column coordinate is given
# def findIndex(row, column, numOfColumns):
#     return row*numOfColumns + column

#Compare the points and points next and return a array with deviated values

def filterPoints(points_ref, points_curr, counterArr, statusArr, threshold=2, marked = False):
    interest_points = []
    new_counterArr = []

    if(not marked):
        points_ref = points_ref[statusArr ==1]
        points_curr = points_curr[statusArr ==1]

        for i in range(len(points_ref)):
            x_diff = points_curr[i][0] - points_ref[i][0]
            y_diff = points_curr[i][1] - points_ref[i][1]

            if( np.sqrt(x_diff**2 + y_diff**2) >= threshold):
                interest_points.append([points_curr[i], points_ref[i]])
                new_counterArr.append(0)

        return interest_points, new_counterArr
    else:

        for i in range(len(points_ref)):

            if (statusArr[i] == 0):
                counterArr[i] = counterArr[i] + 1
            else:
                x_diff = points_curr[i][0][0] - points_ref[i][0][0]
                y_diff = points_curr[i][0][1] - points_ref[i][0][1]

                if (np.sqrt(x_diff ** 2 + y_diff ** 2) >= threshold):
                    counterArr[i] = 0
                else:
                    counterArr[i] = counterArr[i] + 1

            if(counterArr[i] < 3):
                interest_points.append([points_curr[i], points_ref[i]])
                new_counterArr.append(counterArr[i])


        return interest_points, new_counterArr

#Returns interesting points with their reference points and initial counter array
def calculateOpticalFlowRegular(frame, frameRef, points_ref,lkparams):

    points_curr, statusArr, err = cv2.calcOpticalFlowPyrLK(
        frameRef, frame, points_ref, None, **lkparams
    )

    #Filtering points to get the points with possible actual movements
    filteredPoints, counterArr = filterPoints(points_ref, points_curr, None, statusArr)

    interestingPoints_curr = np.array([[[0, 0]]], dtype=np.float32)
    interestingPoints_ref = np.array([[[0, 0]]], dtype=np.float32)
    for points in filteredPoints:
        curr = points[0]
        ref = points[1]
        a, b = curr.ravel()
        c, d = ref.ravel()
        interestingPoints_curr = np.append(interestingPoints_curr, [[[a, b]]], axis=0)
        interestingPoints_ref = np.append(interestingPoints_ref, [[[c, d]]], axis=0)

    interestingPoints_curr = np.delete(interestingPoints_curr, 0, axis=0)
    interestingPoints_ref = np.delete(interestingPoints_ref, 0, axis=0)

    return interestingPoints_curr, interestingPoints_ref, counterArr

#Returns Interesting points with their reference points and updated counter array
def calculateOpticalFlowMarked(frame, frameRef, pointsMarked_ref, counterArr, lkparams):

    pointsMarked_curr, status, err = cv2.calcOpticalFlowPyrLK(
        frameRef, frame, pointsMarked_ref, None, **lkparams
    )

    markedPoints, counterArr = filterPoints(pointsMarked_ref, pointsMarked_curr, counterArr,
                                                status, marked= True)

    interestingPoints_curr = np.array([[[0, 0]]], dtype=np.float32)
    interestingPoints_ref = np.array([[[0, 0]]], dtype=np.float32)
    for points in markedPoints:
        curr = points[0]
        ref = points[1]
        a, b = curr.ravel()
        c, d = ref.ravel()
        interestingPoints_curr = np.append(interestingPoints_curr, [[[a, b]]], axis=0)
        interestingPoints_ref = np.append(interestingPoints_ref, [[[c, d]]], axis=0)

    interestingPoints_curr = np.delete(interestingPoints_curr, 0, axis=0)
    interestingPoints_ref = np.delete(interestingPoints_ref, 0, axis=0)


    return interestingPoints_curr, interestingPoints_ref, counterArr

def __main__():
    videoFile = cv2.VideoCapture(fileName)

    ret, frameRef = videoFile.read()
    while(ret != True):
        ret, frameRef = videoFile.read()
    frameRef_gray = cv2.cvtColor(frameRef, cv2.COLOR_BGR2GRAY)
    frameRef = filterFrame(frameRef)

    regularPoints, markedPoints_ref, counterArr, mask = initialValuesLK(frameRef, pointStep)
    while videoFile.isOpened():
        ret, frameCurr = videoFile.read()
        frameCurr_gray = cv2.cvtColor(frameCurr, cv2.COLOR_BGR2GRAY)
        frameCurr = filterFrame(frameCurr)

        if ret:
            interestingPointsTemp_curr, interestingPointsTemp_ref, counterArrTemp = \
                calculateOpticalFlowRegular(frameCurr_gray, frameRef_gray,
                                            regularPoints, lkparams)

            markedPoints_curr, markedPoints_ref, counterArr = \
                calculateOpticalFlowMarked(frameCurr_gray, frameRef_gray, markedPoints_ref,
                                           counterArr, lkparams)

            for i in range(len(markedPoints_curr)):
                new = markedPoints_curr[i]
                old = markedPoints_ref[i]
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), [0, 255, 0], 1)
                frameCurr = cv2.circle(frameCurr, (a, b), 2, [255, 0, 0], -1)

            # markedPoints_ref = np.concatenate((markedPoints_curr, interestingPointsTemp_curr), axis=0)
            # counterArr = counterArr + counterArrTemp

            frameRef_gray = frameCurr_gray
            markedPoints_ref = np.concatenate((markedPoints_curr, interestingPointsTemp_curr), axis=0)
            counterArr = counterArr + counterArrTemp

            frameCurr = cv2.add(frameCurr, mask)
            cv2.imshow('frame', frameCurr)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    videoFile.release()
    cv2.destroyAllWindows()
    print("Successfully Completed!")


__main__()