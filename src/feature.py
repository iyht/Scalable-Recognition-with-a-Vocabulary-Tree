import cv2
import numpy as np


class FeatureDetector():
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()

    def detect(self, img1, method='SIFT'):
        # setup the SIFT
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Compute the keypoints and its corresponding descriptors of each images
        if(method == 'SIFT'):
            kp1, des1 = self.sift.detectAndCompute(gray1,None)
        elif(method == 'ORB'):
            kp1, des1 = self.orb.detectAndCompute(gray1,None)

        return kp1, des1

    def match(self, kp1, kp2, des1, des2, method):
        '''
        Input:
        - kp1: keypoints from img1  
        - kp2: keypoints from img2  
        - des1: descriptor from img1 
        - des2: descriptor from img2 
        Output:
        - a list of tuple in format e.g., 
        [([kp2.x, kp2.y, 1], [kp1.x, kp1.y, 1], ratio), ..., ()]
        '''
        if method == 'SIFT':
            result = []
            for i in range(len(des1)):
                # Compare each key point from img1 with all the key point from img2
                # to find the smallest_distance, and the second_smallest_distance
                smallest_distance = np.inf
                second_smallest_distance = np.inf
                smallest_j = 0
                for j in range(len(des2)):
                    distance = np.linalg.norm(des1[i] - des2[j])
                    if distance < smallest_distance:
                        second_smallest_distance = smallest_distance
                        smallest_distance = distance
                        smallest_j = j
                ratio = smallest_distance / second_smallest_distance
                # If the the ratio smaller than the threshold, then it means this correspondence is not 
                # reliable. We are free to ignore this pair of matching point.
                if ratio < 0.8:
                    # If the correspondence is reliable, then we add it to the result.
                    result.append((np.append(kp1[i].pt, [1]), \
                                   np.append(kp2[smallest_j].pt, [1]), \
                                   ratio))
        elif method == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            dmatches = sorted(matches, key = lambda x:x.distance)
            src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
            for i in range(len(src_pts)):
                result.append((np.append(src_pts[i].pt, [1]), \
                               np.append(dst_pts[i].pt, [1])))
        return result
    
    def detect_and_match(self, img1, img2, method='SIFT'):
        # setup the SIFT
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute the keypoints and its corresponding descriptors of each images
        if method == 'SIFT':
            kp1, des1 = self.sift.detectAndCompute(gray1,None)
            kp2, des2 = self.sift.detectAndCompute(gray2,None)
            # Filter out the unreliable correspondences
            correspondences = self.match(kp1, kp2, des1, des2, method)
        elif method == 'ORB':
            kp1, des1 = self.orb.detectAndCompute(gray1,None)
            kp2, des2 = self.orb.detectAndCompute(gray2,None)
            correspondences = self.match(kp1, kp2, des1, des2, method)

        return correspondences

    def draw_circle(self, image, kp):
        shape = image.shape
        H = shape[0]
        W = shape[1]
        for i in range(0, len(kp)):
            pt = kp[i].pt
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), int(kp[i].size), (0, 255, 0), 1)
        return image



def filter_match_points(kp1, kp2, des1, des2):
    '''
    Input:
    - kp1: keypoints from img1  
    - kp2: keypoints from img2  
    - des1: descriptor from img1 
    - des2: descriptor from img2 
    Output:
    - a list of tuple in format e.g., 
    [([kp2.x, kp2.y, 1], [kp1.x, kp1.y, 1], ratio), ..., ()]
    '''
    result = []
    for i in range(len(des1)):
        # Compare each key point from img1 with all the key point from img2
        # to find the smallest_distance, and the second_smallest_distance
        smallest_distance = np.inf
        second_smallest_distance = np.inf
        smallest_j = 0
        for j in range(len(des2)):
            distance = np.linalg.norm(des1[i] - des2[j])
            if distance < smallest_distance:
                second_smallest_distance = smallest_distance
                smallest_distance = distance
                smallest_j = j
        ratio = smallest_distance / second_smallest_distance
        # If the the ratio smaller than the threshold, then it means this correspondence is not 
        # reliable. We are free to ignore this pair of matching point.
        if ratio < 0.8:
            # If the correspondence is reliable, then we add it to the result.
            result.append((np.append(kp1[i].pt, [1]), \
                           np.append(kp2[smallest_j].pt, [1]), \
                           ratio))
            # result.append((np.append(kp2[smallest_j].pt, [1]), \
            #                np.append(kp1[i].pt, [1]), \
            #                ratio))
    return result

def SIFT_match_points(img1, img2):
    # setup the SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    # Compute the keypoints and its corresponding descriptors of each images
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    # Filter out the unreliable correspondences
    correspondences = filter_match_points(kp1, kp2, des1, des2)
    return correspondences

def SIFT_match_points_single(img1):
    # setup the SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    # Compute the keypoints and its corresponding descriptors of each images
    kp1, des1 = sift.detectAndCompute(gray1,None)

    return kp1, des1


def draw_circle(image, kp):
    shape = image.shape
    H = shape[0]
    W = shape[1]
    for i in range(0, len(kp)):
        pt = kp[i].pt
        image = cv2.circle(image, (int(pt[0]), int(pt[1])), int(kp[i].size), (0, 255, 0), 1)
    return image

#test_path = '../data/test'
#test_image1 = test_path + '/image_01.jpeg'
#
#img1 = cv2.imread(test_image1)
#kp1, des1 = SIFT_match_points_single(img1)
#import pdb;pdb.set_trace()
#img1_draw = draw_circle(img1, kp1)
#cv2.imshow("img1", img1)
#cv2.imshow("img1_draw", img1_draw)
#cv2.waitKey(0)