import cv2
import numpy as np
from core.config import *

"""
Given two picture representing the same view with a different angle, the function compute the homography matrix from the zenithal_frame to the second frame.
The ORB feature detector is used to find the keypoint and a Brute force matcher is used to associate pair of points.
The pair of points that are not closer enough are discarded (an error distance smaller then 30).
This function will throw an error if less then 4 point are match correctly.
Input:  -zenithal_frame: a picture of a scene taken from a zenithal position.
        -frame2: a picture the same scene that zenithal_frame but from a different point of view.
Output: -M: a homography matrix.
"""
def compute_homography_matrix(zenithal_frame, frame2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(zenithal_frame, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

    # Match the keypoints using their descriptor and sort them with minimal distance (maximal similarity)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Compute the homography using all feature correspondences (we can reduce that for lower computing time)
    end_index = 0
    for match in matches:
        if match.distance >= 30.0:
            break
        end_index += 1

    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches[:end_index] ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches[:end_index] ]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

"""
Given two sets of points of the same size, the function compute the homography matrix from them.
Input:  -src_pts: a set of points being a vector of pairs of float.
        -dst_pts: a set of points being a vector of pairs of float.
Output: -M: a homography matrix, a 3x3 float matrix.
"""
def compute_homography_from_point(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

"""
Given a point and a homography matrix, the function compute the transform of the point by the matrix.
Input:  -src_pt: a point being a vector of 3 float with always 1 as third digit.
        -M: a homography matrix, a 3x3 float matrix.
Output: -dst_pt: a point being a pair float.
"""
def compute_destination_point(src_pt, M):
    dst_pt = np.dot(M, src_pt)
    dst_pt = dst_pt/dst_pt[2]

    return dst_pt[0:-1]