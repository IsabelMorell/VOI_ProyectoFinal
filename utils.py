import os
import cv2
import numpy as np

def create_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

def save_images(img, img_name: str, folder_path: str = "."):
    if img_name[-4:] != ".jpg":
        img_name = f"{img_name}.jpg"
    img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(img_path, img)

def show_image(img) -> None:
    cv2.imshow("Chessboard Images", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_max_suppression(img, theta):
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
    
def color_segmentation(img, color):
    return img_segmented