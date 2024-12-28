from typing import List, Tuple
from utils import *

import numpy as np
import constants as cte

import cv2
import copy
import glob
import os


def find_chessboard_corners(imgs):
    corners = [cv2.findChessboardCorners(img, (cte.WIDTH, cte.HEIGHT), None) for img in imgs]
    return corners

def safe_corners(folder_path: str, img_name: str, imgs, corners) -> None:
    os.makedirs(folder_path, exist_ok=True)
    for i in range(len(imgs)):
        print(corners[0])
        if corners[0]:
            cv2.drawChessboardCorners(imgs[i], (cte.WIDTH, cte.HEIGHT), corners[i][1], corners[i][0])
            show_image(imgs[i])
            if i < 10:
                path = os.path.join(folder_path, f"{img_name}_0{i}.jpg")
            else:
                path = os.path.join(folder_path, f"{img_name}_{i}.jpg")
            save_images(path, imgs[i])
        else:
            print("Esquinas no encontradas")

def get_chessboard_points(chessboard_shape, dx, dy):
    matriz = []
    # SE HA CAMBIADO: i SON COLUMNAS Y j SON FILAS
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            matriz.append(np.array([i*dx, j*dy, 0]))
    return np.array(matriz, dtype=np.float32)

def get_corners_and_chessboards_points(imgs, corners):
    imgs_copy = copy.deepcopy(imgs)
    valid_corners = []
    chessboard_points = []
    for corner in zip(imgs_copy,corners):
        if corner[0]:
            valid_corners.append(corner[1])
            chessboard_points.append(get_chessboard_points((cte.WIDTH, cte.HEIGHT), cte.DX, cte.DY))
        else:
            valid_corners.append(0)
            chessboard_points.append(0)
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    chessboard_points = np.asarray(chessboard_points, dtype=np.float32)
    return chessboard_points, valid_corners
    
def calibrate_camera():
    imgs_path = glob.glob('./data/chessboard_frames/*.jpg')
    folder = "./data/chessboard_corners"
    img_name = "chessboard_corners"
    imgs = load_images(imgs_path)
    # Detectamos esquinas
    corners = find_chessboard_corners(imgs)
    safe_corners(folder, img_name, imgs, corners)
    chessboard_points, valid_corners = get_corners_and_chessboards_points(imgs, corners)
    # Calibramos la cámara
    height, width, _ = imgs[0].shape
    for i in range(len(imgs)):
        rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points[i], valid_corners[i], (height, width), None, None)

if __name__=="__main__":
    calibrate_camera()