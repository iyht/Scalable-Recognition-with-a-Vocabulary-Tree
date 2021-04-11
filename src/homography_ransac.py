from sift import *
import cv2
import numpy as np
from skimage import io
from skimage import draw
from skimage import transform
import random
from matplotlib import pyplot as pl


def homography(correspondences):
    n = len(correspondences) # number of points
    A = np.zeros((2*n, 9))
    for i in range(len(correspondences)):
        pt1 = correspondences[i][0]
        pt2 = correspondences[i][1]
        A[2*i, 0] = pt1[0]
        A[2*i, 1] = pt1[1]
        A[2*i, 2] = 1
        A[2*i, 6] = -pt1[0]*pt2[0]
        A[2*i, 7] = -pt1[0]*pt2[1]
        A[2*i, 8] = -pt2[0]
        A[2*i+1, 3] = pt1[0]
        A[2*i+1, 4] = pt1[1]
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -pt2[1]*pt1[0]
        A[2*i+1, 7] = -pt2[1]*pt1[1]
        A[2*i+1, 8] = -pt2[1]
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    H = Vh[-1].reshape((3,3))
    return H

def num_round_needed(p, k, P):
    '''
    find the number of rounds needed for RANSAC to have P chance to success.
    Inputs:
    - p: the probability that the sample match is inlier
    - k: number of matches we need for one round
    - P: the probability success after S rounds
    '''
    S = np.log(1 - P) / np.log(1 - p**k)
    return int(S)

def RANSAC_find_optimal_Homography(correspondences):
    '''
    Input:
    - correspondences: a list of tuple that stores the correspondences 
        e.g., [([kp2.x, kp2.y, 1], [kp1.x, kp1.y, 1]) ... ([],[]) ]
    Output:
    - a 3x3 homography matrix
    '''
    # project matched point and compute the difference
    optimal_H = None
    optimal_inliers = 0
    num_rounds = num_round_needed(0.15, 4, 0.98)
    for i in range(num_rounds):
        # random sample 4 keypoint pairs
        sample_corr = random.sample(correspondences, 4)
        # compute the homography
        H = homography(sample_corr)
        num_inliers = 0
        for pair in correspondences:
            pt1 = pair[0]
            pt2 = pair[1]
            projected_pt1 = H @ pt1
            projected_pt1 /= projected_pt1[2]
            loss = np.linalg.norm(pt2 - projected_pt1)
            if loss < 4:
                num_inliers += 1
        if num_inliers > optimal_inliers:
            optimal_H = H
            optimal_inliers = num_inliers
    return optimal_H


def visualize_homograpy(img1, img2, H):
    # calculate the affine transformation matrix
    height = img1.shape[0]
    width = img1.shape[1]
    # define the reference points
    point1 = np.array([0, 0, 1])
    point2 = np.array([width-1, 0, 1])
    point3 = np.array([width-1, height-1, 1])
    point4 = np.array([0, height-1, 1])
    # transfer the points with affine transformation to get the new point on img2
    new_point1 = H @ point1
    new_point1 = new_point1/new_point1[2]
    new_point2 = H @ point2
    new_point2 = new_point2/new_point2[2]
    new_point3 = H @ point3
    new_point3 = new_point3/new_point3[2]
    new_point4 = H @ point4
    new_point4 = new_point4/new_point4[2]
    # draw the line
    cv2.line(img2, (int(new_point1[0]), int(new_point1[1])), (int(new_point2[0]), int(new_point2[1])), (255,0,0), 3)
    cv2.line(img2, (int(new_point2[0]), int(new_point2[1])), (int(new_point3[0]), int(new_point3[1])), (255,0,0), 3)
    cv2.line(img2, (int(new_point3[0]), int(new_point3[1])), (int(new_point4[0]), int(new_point4[1])), (255,0,0), 3)
    cv2.line(img2, (int(new_point4[0]), int(new_point4[1])), (int(new_point1[0]), int(new_point1[1])), (255,0,0), 3)
    cv2.imwrite('homo_visiual.png', img2)
    return img2


test_path = '../data/test'
cover_path = '../data/DVDcovers'
# cover = cover_path + '/reference.png'
cover = cover_path + '/matrix.jpg'
test = test_path + '/image_07.jpeg'

cover = cv2.imread(cover)
cv2.imshow("cover", cover)
cv2.waitKey(0)
# test = cv2.imread('./test.png')
test = cv2.imread(test)
cv2.imshow("test", test)
cv2.waitKey(0)

######################
# orb = cv2.ORB_create()

# gray2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# gray1 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)

# ## Find the keypoints and descriptors with ORB
# kpts1, descs1 = orb.detectAndCompute(gray1,None)
# kpts2, descs2 = orb.detectAndCompute(gray2,None)

# ## match descriptors and sort them in the order of their distance
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descs1, descs2)
# dmatches = sorted(matches, key = lambda x:x.distance)

# ## extract the matched keypoints
# src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
# dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

## find homography matrix and do perspective transform
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# h,w = cover.shape[:2]
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,M)

# ## draw found regions
# test = cv2.polylines(test, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
# cv2.imshow("found", test)

# ## draw match lines
# res = cv2.drawMatches(cover, kpts1, test, kpts2, dmatches[:20],None,flags=2)

# cv2.imshow("orb_match", res);

# cv2.waitKey();cv2.destroyAllWindows()
######################

correspondences = SIFT_match_points(cover, test)
optimal_H = RANSAC_find_optimal_Homography(correspondences)

# src_pts = np.float32([ t[0][:2] for t in correspondences ]).reshape(-1,1,2)
# dst_pts = np.float32([ t[1][:2] for t in correspondences ]).reshape(-1,1,2)
# optimal_H_build, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1)
# import pdb;pdb.set_trace()
# h,w = cover.shape[:2]
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,optimal_H)
# test = cv2.polylines(test, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
# cv2.imshow("found", test)

## draw match lines
# res = cv2.drawMatches(cover, kpts1, test, kpts2, dmatches[:20],None,flags=2)

# cv2.imshow("orb_match", res);
visualize_homograpy(cover, test, optimal_H)

cv2.waitKey();cv2.destroyAllWindows()