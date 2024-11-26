from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import imageio
import cv2
import copy
import glob
import os

def find_chessboard_corners(imgs: List[np.array()], dimensions: Tuple[int,int]):
    corners = [cv2.findChessboardCorners(img, (8, 6), None) for img in imgs]
    return corners

def refine_corners(corners):
    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # TODO To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]
    return corners_refined


