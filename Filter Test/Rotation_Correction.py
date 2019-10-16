from __future__ import print_function
import cv2
import numpy as np
from vidstab.VidStab import VidStab


MAX_FEATURES = 1500
GOOD_MATCH_PERCENT = 0.05


def alignImages(referenceImg, image, MAX_FEATURES = MAX_FEATURES, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT):

    # Convert images to grayscale
    im1Gray = referenceImg
    im2Gray = image

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
    height, width = referenceImg.shape
    if (type(h) != np.ndarray):
        print("Homography matrix is none")
        return image

    im1Reg = cv2.warpPerspective(im2Gray, h, (width, height), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    return im1Reg

