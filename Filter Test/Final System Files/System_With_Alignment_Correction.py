from Filter_Frame import *
from LK_Method import *
from Rotation_Correction import *
from vidstab.VidStab import VidStab


fileName = "/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/CleanNew.mp4"

videoFile = cv2.VideoCapture(fileName)

stabilizer = VidStab()
cap = cv2.VideoCapture(fileName)

isFirst = True

ret, alignedFrameRef = cap.read()
while(ret != True):
    ret, frameRef = videoFile.read()
alignedFrameRef = cv2.cvtColor(alignedFrameRef, cv2.COLOR_BGR2GRAY)
alignedFrameRef = filterFrame(alignedFrameRef)

mask, numOfPoints, pointsInterest_count, pointsInterest_curr, points = initializeMethod(alignedFrameRef)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = filterFrame(frame)

        alignedFrameCurr = alignImages(alignedFrameRef, frame)
        alignedFrameCurr = stabilizer.stabilize_frame(input_frame=alignedFrameCurr, smoothing_window=30)

        if(isFirst):
            isFirst = False
        else:
            frameGray, points_nxt, pointsInterest_nxt, pointsInterest_count = calculateOpticalFlow(alignedFrameCurr,
                                                                                                   alignedFrameRef, points, pointsInterest_curr,
                                                                                                   pointsInterest_count, mask, lk_params)

            points = points_nxt
            pointsInterest_curr = pointsInterest_nxt
            cv2.imshow('frame', frameGray)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        alignedFrameRef = alignedFrameCurr



cap.release()
cv2.destroyAllWindows()
print("Successfully Completed!")
