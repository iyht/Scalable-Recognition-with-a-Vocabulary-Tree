from feature import *
import cv2
import numpy as np
from skimage import io
from skimage import draw
from skimage import transform
import random
from matplotlib import pyplot as pl
random.seed(999)



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

def RANSAC_find_optimal_Homography(correspondences, num_rounds=None):
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
    num_rounds = num_rounds if num_rounds != None else num_round_needed(0.15, 4, 0.95)
    for i in range(num_rounds):
        # random sample 4 keypoint pairs
        sample_corr = random.sample(correspondences, 4)
        # compute the homography
        H = homography(sample_corr)
        num_inliers = 0
        # for pair in correspondences[:100]:
        for pair in correspondences:
            pt1 = pair[0]
            pt2 = pair[1]
            projected_pt1 = H @ pt1
            projected_pt1 /= projected_pt1[2]
            loss = np.linalg.norm(pt2 - projected_pt1)
            if loss <= 20:
                num_inliers += 1
        if num_inliers > optimal_inliers:
            optimal_H = H
            optimal_inliers = num_inliers
    return optimal_inliers, optimal_H


def visualize_homograpy(img1, img2, H):
    # calculate the affine transformation matrix
    # h = img1.shape[0]
    # w = img1.shape[1]
    h,w = img1.shape[:2]

    # define the reference points
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # transfer the points with affine transformation to get the new point on img2
    dst = cv2.perspectiveTransform(pts,H)
    result = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite("result.png", result)

    return result

