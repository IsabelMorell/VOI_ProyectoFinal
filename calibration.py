from typing import List, Tuple
from utils import *

import matplotlib.pyplot as plt
import numpy as np

import imageio
import cv2
import copy
import glob
import os

# Number of inner corners of the image
WIDTH = 12
HEIGHT = 8

# Constants needed for refine_corners
ZERO_ZONE = (-1, -1)
NUM_ITERATIONS = 30
ACCURACY = 0.01


def find_chessboard_corners(imgs: List[np.array()], dimensions: Tuple[int,int]):
    corners = [cv2.findChessboardCorners(img, (WIDTH, HEIGHT), None) for img in imgs]
    return corners

def refine_corners(imgs, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, NUM_ITERATIONS, ACCURACY)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    corners_copy = copy.deepcopy(corners)
    corners_refined = [cv2.cornerSubPix(img, corner[1], (WIDTH, HEIGHT), ZERO_ZONE, criteria) if corner[0] else [] for img, corner in zip(imgs_gray, corners_copy)]
    return corners_refined

def safe_corners(folder_path: str, img_name: str, imgs, corners):
    os.makedirs(folder_path, exist_ok=True)
    for i in range(len(imgs)):
        if corners[i][0]:
            cv2.drawChessboardCorners(imgs[i], (WIDTH, HEIGHT), corners[i][1], corners[i][0])
            show_image(imgs[i])
            if i < 10:
                path = os.path.join(folder_path, f"{img_name}_0{i}.jpg")
            else:
                path = os.path.join(folder_path, f"{img_name}_{i}.jpg")
            write_image(path, imgs[i])
        else:
            print("Esquinas no encontradas")

def get_chessboard_points(chessboard_shape, dx, dy):
    N = chessboard_shape[0]*chessboard_shape[1]
    matriz = []
    for i in range(chessboard_shape[0]):
        for j in range(chessboard_shape[1]):
            matriz.append(np.array([i*dx, j*dy, 0]))
    return np.array(matriz, dtype=np.float32)


def get_corners_and_chessboards_points(imgs, corners):
    imgs_copy = copy.deepcopy(imgs)
    valid_corners = []
    chessboard_points = []

    for img, corner in zip(imgs_copy, corners):
        if corner[0]:
            valid_corners.append(corner[1])
            dx = 30
            dy = 30
            chessboard_points.append(get_chessboard_points((WIDTH, HEIGHT), dx, dy))
            
    # Convert list to numpy array
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    chessboard_points = np.asarray(chessboard_points, dtype=np.float32)
    
def calibrate_camera():
    folder = 
    height, width, _ = imgs[0].shape
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, , None, None)

