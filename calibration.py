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

# 
DX = 30
DY = 30

def load_images(filenames):
    return [imageio.v2.imread(filename) for filename in filenames]

def show_image(img) -> None:
    cv2.imshow("Chessboard Images", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(path, img) -> None:
    cv2.imwrite(path, img)


def find_chessboard_corners(imgs):
    corners = [cv2.findChessboardCorners(img, (WIDTH, HEIGHT), None) for img in imgs]
    corners_refined = refine_corners(corners)
    return corners_refined

def refine_corners(imgs, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, NUM_ITERATIONS, ACCURACY)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    corners_copy = copy.deepcopy(corners)
    corners_refined = [cv2.cornerSubPix(img, corner[1], (WIDTH, HEIGHT), ZERO_ZONE, criteria) if corner[0] else [] for img, corner in zip(imgs_gray, corners_copy)]
    return corners_refined

def safe_corners(folder_path: str, img_name: str, imgs, corners) -> None:
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
    # SE HA CAMBIADO: i SON COLUMNAS Y j SON FILAS
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            matriz.append(np.array([i*dx, j*dy, 0]))
    return np.array(matriz, dtype=np.float32)


# def get_corners_and_chessboards_points(imgs, corners):
#     imgs_copy = copy.deepcopy(imgs)
#     valid_corners = []
#     chessboard_points = []

#     for corner in zip(imgs_copy,corners):
#         if corner[0]:
#             valid_corners.append(corner[1])
#             dx = 30
#             dy = 30
#             chessboard_points.append(get_chessboard_points((WIDTH, HEIGHT), dx, dy))
#         else:
#             valid_corners.append(0)
#             chessboard_points.append(0)
            
#     # Convert list to numpy array
#     valid_corners = np.asarray(valid_corners, dtype=np.float32)
#     chessboard_points = np.asarray(chessboard_points, dtype=np.float32)
#     return chessboard_points, valid_corners
    
def calibrate_camera():
    imgs_path = glob.glob('data/*.jpg')
    folder = ""
    img_name = ""
    imgs = load_images(imgs_path)
    # Detectamos esquinas
    corners = find_chessboard_corners(imgs)  # corners refined
    safe_corners(folder, img_name, imgs, corners)
    # Nos quedamos con los corners válidos
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    chessboard_points = get_chessboard_points((WIDTH, HEIGHT), DX, DY)
    # height, width, _ = imgs[0].shape
    for i in range(len(imgs)):
        rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points[i], valid_corners[i], imgs[i].shape[:2], None, None)

