import cv2
import matplotlib.pyplot as plt
import numpy as np

def ConvertImageColorspace(img1, img2):
    img1_cvt = img1
    img2_cvt = img2
    
    if img1.ndim > 2 and img1.shape[2] == 3:
        img1_cvt = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim > 2 and img2.shape[2] == 3:
        img2_cvt = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    return img1_cvt, img2_cvt

def GenerateKeypointsDescriptorSIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def MatchDescriptorsSIFT(des1, des2, maximal_dist):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    
    good = []
    for m in matches:
        if (m[0].distance < maximal_dist*m[1].distance):
            good.append(m)
    matches = np.asanyarray(good)
    return matches
    
def GenerateHomographyTform(kp1, kp2, matches, minimal_matches, maximal_dist):
    if len(matches[:,0]) < minimal_matches:
        raise AssertionError('not enough matches')
        
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, maximal_dist)
    return H
    
def CalculateNewBounds(img, H):
    start = [[0,0,1]]
    end = [[img.shape[0], img.shape[1],1]]
    startTform = np.matmul(H, start)
    endTform = np.matmul(H, end)
    
    startTformFix = [[startTform[0] / startTform[2], startTform[1] / startTform[2]]]
    endTformFix = [[endTform[0] / endTform[2], endTform[1] / endTform[2]]]
    return startTformFix, endTformFix
    
def Stitch(img1, img2, H):
    
