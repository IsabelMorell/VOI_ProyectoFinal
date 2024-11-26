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


# hola