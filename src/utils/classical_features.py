"""
Classical feature extraction methods for person Re-ID.
Includes HOG, color histograms, and SIFT descriptors.
"""

import cv2
import numpy as np

def extract_hog_features(img):
    """
    Extracts HOG descriptors using OpenCV.
    Args:
        img: RGB image (numpy array).
    Returns:
        1D feature vector.
    """
    win_size = (64, 128) if img.shape[0] == 128 else (128, 256)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    img_resized = cv2.resize(img, win_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    hist = hog.compute(img_gray)
    return hist.flatten()

def extract_color_hist(img, bins=(8, 8, 8)):
    """
    Extracts a 3D color histogram in HSV space.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_sift_features(img):
    """
    Extracts SIFT keypoints and descriptors.
    Note: SIFT returns a variable number of descriptors.
    """
    sift = cv2.SIFT_create()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, des = sift.detectAndCompute(img_gray, None)
    return des # Shape: (num_keypoints, 128)