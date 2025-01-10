import os
import cv2
import numpy as np
from typing import List

def create_folder(folder_path: str) -> None:
    """
    Creates a new folder with path folder_path if it doesn´t exist 

    Args:
        folder_path (str): path to the new folder
    """
    os.makedirs(folder_path, exist_ok=True)

def load_images(filenames: List[str]) -> List[np.ndarray]:
    """
    Load the images whose names are in the List filenames

    Args:
        filenames (List[str]): list with the file names

    Returns
        List[np.ndarray]: images corresponding to the file names
    """
    return [cv2.imread(filename) for filename in filenames]

def save_images(img: np.ndarray, img_name: str, folder_path: str = ".") -> None:
    """
    Saves an images with a specific name and in a specific folder

    Args:
        img (np.ndarray): image to save
        img_name (str): name of the image to save
        folder_path (str): path to the folder where the image will be saved
    """
    if img_name[-4:] != ".jpg":
        img_name = f"{img_name}.jpg"
    img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(img_path, img)

def show_image(img: np.ndarray, img_name: str = "Image") -> None:
    """
    Shows an image with a specific name

    Args:
        img (np.ndarray): image to show
        img_name (str): name of the image to show
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression(img: np.ndarray, theta: np.ndarray):
    """
    Applies Non-Maximum Suppression (NMS) to an image to thin edges by suppressing non-maximum gradient values.

    Args:
        img (np.ndarray): original image
        theta (np.ndarray): array of gradient directions (in radians) corresponding to each pixel of the image

    Returns:
        np.ndarray: array of the same shape as the input image, with non-maximum gradients suppressed
    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)

    # converting radians to degree
    angle = theta * 180. / np.pi    # max -> 180, min -> -180
    angle[angle < 0] += 180         # max -> 180, min -> 0

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = img[i, j-1]
                q = img[i, j+1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                r = img[i-1, j+1]
                q = img[i+1, j-1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                r = img[i-1, j]
                q = img[i+1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                r = img[i+1, j+1]
                q = img[i-1, j-1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0
    return Z
    