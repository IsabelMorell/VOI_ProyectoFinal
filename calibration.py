from typing import List, Tuple
from utils import *

import numpy as np
import constants as cte

import cv2
import copy
import glob
import os


def find_chessboard_corners(imgs: List[np.ndarray]) -> List[Tuple[bool, cv2.typing.MatLike]]:
    """
    Finds the corners of the images in imgs

    Args:
        imgs (List[np.ndarray]): original images

    Returns:
        List[Tuple[bool, cv2.typing.MatLike]]: corners of the images
    """
    corners = [cv2.findChessboardCorners(img, (cte.WIDTH, cte.HEIGHT), None) for img in imgs]
    return corners

def save_corners(folder_path: str, img_name: str, imgs: List[np.ndarray], corners: List[Tuple[bool, cv2.typing.MatLike]]) -> None:
    """
    Saves images with the corners identified in the folder folder_path as img_name

    Args:
        folder_path (str): folder where the images are saved
        img_name (str): name of the image that is going to be saved
        imgs (List[np.ndarray]): original images
        corners (List[Tuple[bool, cv2.typing.MatLike]]): corners identified of the original images
    """
    os.makedirs(folder_path, exist_ok=True)
    for i in range(len(imgs)):
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

def get_chessboard_points(chessboard_shape: Tuple[int, int], dx: int, dy: int) -> np.ndarray:
    matriz = []
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            matriz.append(np.array([i*dx, j*dy, 0]))
    return np.array(matriz, dtype=np.float32)

def get_corners_and_chessboards_points(imgs: List[np.ndarray], corners: List[Tuple[bool, cv2.typing.MatLike]]) -> Tuple[np.ndarray, np.ndarray]:
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
    
def calibrate_camera() -> None:
    imgs_path = glob.glob('./data/chessboard_frames/*.jpg')
    folder = "./data/chessboard_corners"
    img_name = "chessboard_corners"
    imgs = load_images(imgs_path)
    # Detectamos esquinas
    corners = find_chessboard_corners(imgs)
    save_corners(folder, img_name, imgs, corners)
    chessboard_points, valid_corners = get_corners_and_chessboards_points(imgs, corners)
    # Calibramos la cámara
    height, width, _ = imgs[0].shape
    for i in range(len(imgs)):
        rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points[i], valid_corners[i], (height, width), None, None)

if __name__=="__main__":
    calibrate_camera()