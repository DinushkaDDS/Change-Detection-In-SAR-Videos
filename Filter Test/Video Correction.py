from __future__ import print_function
import cv2
import numpy as np
from vidstab.VidStab import VidStab


MAX_FEATURES = 1500
GOOD_MATCH_PERCENT = 0.05

fileName = "/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/CleanNew.mp4"


def alignImages(referenceImg, image):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(referenceImg, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    if(type(descriptors2)!= type(descriptors1)):
        print(type(descriptors1) , type(descriptors2))
        print("feature descriptions are not matching by their type")
        return image

    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt


    if(len(points1)==0 or len(points2)==0):
        print(len(matches))
        print("number of points are zero")
        return image

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = referenceImg.shape
    if (type(h) != np.ndarray):
        print("Homography matrix is none")
        return image

    im1Reg = cv2.warpPerspective(image, h, (width, height), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    return im1Reg

###########################################################################################################
# Start of the main program

videoFile = cv2.VideoCapture(fileName)

# Capture the first reference frame
ret, frameRef = videoFile.read()
while(ret != True):
    ret, frameRef = videoFile.read()

stabilizer = VidStab()

# Read until video is completed
while videoFile.isOpened():

    # Capturing the continuing frames
    ret, frame = videoFile.read()
    if ret:
        alignedFrame = alignImages(frameRef, frame)
        stabilized_frame = stabilizer.stabilize_frame(input_frame=alignedFrame, smoothing_window=30)
        cv2.imshow('Frame', stabilized_frame)
        frameRef = alignedFrame

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# Cleaning up the memory
videoFile.release()
cv2.destroyAllWindows()