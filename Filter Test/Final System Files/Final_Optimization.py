from Filter_Frame import *
import cv2
import numpy as np

fileName = "/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/video1.mp4"

# Parameters for Lucas Kanade optical flow
lkparams = dict(winSize=(5, 5),
                 maxLevel=0,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.05))
pointStep = 25

params = cv2.SimpleBlobDetector_Params()
# Change thresholds for intensity. hence the blobs are dark thresholds should be low
params.minThreshold = 0
params.maxThreshold = 125

# Filter by Area.
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
# Filter by Inertia.
params.filterByInertia = True
params.minInertiaRatio = 0.1
params.maxInertiaRatio = 0.9


detector = cv2.SimpleBlobDetector_create(params)
MagnitudeAVG = 0
TanAVG = 0

#Generate Initial values for the lk method
def initialValuesLK(initialFrame, pointStep):

    regularPoints = np.array([[[0, 0]]], dtype=np.float32)
    markedPoints = np.array([[[0, 0]]], dtype=np.float32)
    numOfRows= 0
    for i in range(0, initialFrame.shape[0], pointStep):
        numOfRows = numOfRows + 1
        for j in range(0, initialFrame.shape[1], pointStep):
            regularPoints = np.append(regularPoints, [[[np.float32(j), np.float32(i)]]], axis=0)

    counterArr = [0]

    regularPoints = np.delete(regularPoints, 0, axis=0)
    markedPoints = np.delete(markedPoints, 0, axis=0)
    #A mask image for drawing purposes
    mask = np.zeros_like(initialFrame)

    numOfColumns = int(len(regularPoints)/numOfRows)
    # print(numOfRows, numOfColumns, len(regularPoints)-1, numOfColumns*numOfRows)
    return regularPoints, markedPoints, counterArr, mask, numOfRows, numOfColumns

#Calculate tan value and magnitude of change of a point
def findTanAndMag(index, pointsCurr, pointsRef):
    if(pointsCurr.ndim == 3 ):

        x_diff = abs(pointsCurr[index][0][0] - pointsRef[index][0][0])
        y_diff = abs(pointsCurr[index][0][1] - pointsRef[index][0][1])
    else:

        x_diff = abs(pointsCurr[index][0] - pointsRef[index][0])
        y_diff = abs(pointsCurr[index][1] - pointsRef[index][1])

    if(x_diff!=0):
        tanValue = np.arctan(y_diff/x_diff)
    else:
        tanValue = np.pi/2

    magnitude = x_diff**2 + y_diff**2
    return tanValue, magnitude

#This function ignores the windowSize number of rows and columns from the frame edges
def checkThreshold(index, pointsCurr, pointsRef, numOfRows, numOfColumns, windowSize = 1, threshold= 0.5):

    row, column = findRowAndCol(index, numOfRows, numOfColumns)

    if(row < windowSize or column < windowSize or (numOfRows -1 -row)<windowSize or (numOfColumns-1-column)<windowSize):
        return

    tanSum = 0
    magnitudeSum = 0

    for r in range(row - windowSize, row + windowSize+1 ):
        for c in range(column - windowSize, column + windowSize+1):
            tempIndex = findIndex(r,c,numOfColumns)
            tanVal , magnitude = findTanAndMag(tempIndex, pointsCurr, pointsRef)
            tanSum = tanSum + tanVal
            magnitudeSum = magnitudeSum + magnitude

    totalPoints = (2*windowSize + 1)**2
    avgTan = tanSum/totalPoints
    avgMag = magnitudeSum/totalPoints
    currTan, currMag = findTanAndMag(index, pointsCurr, pointsRef)

    global TanAVG
    global MagnitudeAVG

    if(TanAVG !=0):
        TanAVG = (TanAVG + avgTan)/2
    elif(TanAVG == 0):
      TanAVG = avgTan

    if (MagnitudeAVG != 0):
        MagnitudeAVG = (MagnitudeAVG + avgMag) / 2
    elif (MagnitudeAVG == 0):
        MagnitudeAVG = avgMag

    # print(MagnitudeAVG, TanAVG)

    if(avgTan == 0 or avgMag == 0):
        return False
    varTan = abs(avgTan-currTan)/avgTan
    varMag = abs(avgMag-currMag)/avgMag


    if(varMag>threshold and varTan>threshold):
        print(avgTan, avgMag)
        return True
    else:
        return False

#Function to get the matrix coordinate of the given index
def findRowAndCol(index, numofRows, numofColumns):
    row = 0
    column = 0
    while(index != (numofColumns*row + column)):

        column = column + 1
        if(column>= numofColumns):
            column = 0
            row = row + 1

        if(row> numofRows):
            return None

    return row, column

#Function to get the index when row,column coordinate is given
def findIndex(row, column, numOfColumns):
    return (row*numOfColumns + column)

#Compare the points and points next and return a array with deviated values
def filterPoints(points_ref, points_curr, counterArr, statusArr, numRows, numColumns, threshold=(MagnitudeAVG, TanAVG), marked = False):
    interest_points = []
    new_counterArr = []

    if(not marked):

        for i in range(len(points_ref)):
            if (checkThreshold(i, points_curr, points_ref, numRows, numColumns) and statusArr[i]==1):
                interest_points.append([points_curr[i], points_ref[i]])
                new_counterArr.append(0)

        return interest_points, new_counterArr

    else:

        for i in range(len(points_ref)):

            if (statusArr[i] == 0):
                counterArr[i] = counterArr[i] + 1
            else:

                x_diff = abs(points_curr[i][0][0] - points_ref[i][0][0])
                y_diff = abs(points_curr[i][0][1] - points_ref[i][0][1])

                if (x_diff != 0):
                    tanValue = np.arctan(y_diff / x_diff)
                else:
                    tanValue = np.pi / 2

                #improve this Part of Thresholding
                if (x_diff ** 2 + y_diff ** 2 >= threshold[0] and tanValue>=threshold[1]):
                    counterArr[i] = 0
                else:
                    counterArr[i] = counterArr[i] + 1

            if(counterArr[i] < 3):
                interest_points.append([points_curr[i], points_ref[i]])
                new_counterArr.append(counterArr[i])
        return interest_points, new_counterArr

#Returns interesting points with their reference points and initial counter array
def calculateOpticalFlowRegular(frame, frameRef, points_ref,lkparams, numRows, numColumns):

    points_curr, statusArr, err = cv2.calcOpticalFlowPyrLK(
        frameRef, frame, points_ref, None, **lkparams
    )

    #Filtering points to get the points with possible actual movements
    filteredPoints, counterArr = filterPoints(points_ref, points_curr, None, statusArr, numRows, numColumns)

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
def calculateOpticalFlowMarked(frame, frameRef, pointsMarked_ref, counterArr, lkparams, numRows, numColumns):

    pointsMarked_curr, status, err = cv2.calcOpticalFlowPyrLK(
        frameRef, frame, pointsMarked_ref, None, **lkparams
    )

    markedPoints, counterArr = filterPoints(pointsMarked_ref, pointsMarked_curr, counterArr,
                                                status, numRows, numColumns, marked= True)

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

#Function to points inside the blobs
def findNeibouringBlob(point, blobPoints, blobSizes):

    x, y = point

    for i in range(len(blobPoints)):
        x_b, y_b = blobPoints[i]
        radius = blobSizes[i]/2

        if(abs(x-x_b)<=radius and abs(y-y_b)<=radius):
            return True
    else:
        return False

#Function to get distance between 2 points
def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

#Function to merge near points
def mergePoints(points, counters, threshold=pointStep):

    retPoints = np.array([[[0, 0]]], dtype=np.float32)
    retCounter = []

    d2 = threshold * threshold
    n = len(counters)
    taken = [False] * n

    for i in range(n):
        if not taken[i]:
            count = 1
            x,y = points[i][0]
            counter = counters[i]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i][0], points[j][0]) < d2:
                    x = x + points[j][0][0]
                    y = y + points[j][0][1]
                    counter = counter + counters[j]
                    count+=1
                    taken[j] = True
            x = x/count
            y = y/count
            counter = int(round(counter/count))
            retPoints =  np.append(retPoints, [[[np.float32(x), np.float32(y)]]], axis=0)
            retCounter.append(counter)

    retPoints = np.delete(retPoints, 0, axis=0)

    return retPoints, retCounter

#Main Function to run the Algorithm
def __main__():
    videoFile = cv2.VideoCapture(fileName)

    ret, frameRef = videoFile.read()
    while(ret != True):
        ret, frameRef = videoFile.read()
    frameRef_gray = cv2.cvtColor(frameRef, cv2.COLOR_BGR2GRAY)
    # frameRef = filterFrame(frameRef)
    # frameCurr_gray = cv2.equalizeHist(frameRef_gray)

    regularPoints, markedPoints_ref, counterArr, mask, numRows, numColumns = initialValuesLK(frameRef, pointStep)
    while videoFile.isOpened():
        ret, frameCurr = videoFile.read()
        frameCurr_gray = cv2.cvtColor(frameCurr, cv2.COLOR_BGR2GRAY)

        # frameCurr = filterFrame(frameCurr)

        if ret:
            interestingPointsTemp_curr, interestingPointsTemp_ref, counterArrTemp = \
                calculateOpticalFlowRegular(frameCurr_gray, frameRef_gray,
                                            regularPoints, lkparams, numRows, numColumns)

            markedPoints_curr, markedPoints_ref, counterArr = \
                calculateOpticalFlowMarked(frameCurr_gray, frameRef_gray, markedPoints_ref,
                                           counterArr, lkparams, numRows, numColumns)

            blobFrame = cv2.equalizeHist(frameCurr_gray)
            keypoints = detector.detect(blobFrame)
            sizes = [x.size for x in keypoints]
            keypoints = cv2.KeyPoint_convert(keypoints)

            for i in range(len(markedPoints_curr)):

                if(findNeibouringBlob(markedPoints_curr[i][0], keypoints, sizes)):

                    new = markedPoints_curr[i]
                    old = markedPoints_ref[i]
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # mask = cv2.line(mask, (a, b), (c, d), [0, 255, 0], 1)
                    frameCurr = cv2.circle(frameCurr, (a, b), 15, [0,255, 0], -1)

            frameRef_gray = frameCurr_gray
            markedPoints_ref = np.concatenate((markedPoints_curr, interestingPointsTemp_curr), axis=0)
            counterArr = counterArr + counterArrTemp

            a, b = mergePoints(markedPoints_ref, counterArr)

            markedPoints_ref = a
            counterArr = b

            # print(TanAVG, MagnitudeAVG)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            frameCurr = cv2.add(frameCurr, mask)
            cv2.imshow('frame', frameCurr)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                continue

    videoFile.release()
    cv2.destroyAllWindows()
    print("Successfully Completed!")

__main__()