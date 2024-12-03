import os
import cv2
import imageio
import glob
import os
import time

import numpy as np

from picamera2 import Picamera2
from typing import List
from utils_lab2 import non_max_suppression, get_hsv_color_ranges


def load_images(filenames):
    return [imageio.v2.imread(filename) for filename in filenames]

def show_image(img) -> None:
    cv2.imshow("Chessboard Images", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_images(img, img_name: str, folder_path: str = "."):
    if img_name[-4:] != ".jpg":
        img_name = f"{img_name}.jpg"
    img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(img_path, img)

# imgs_path = glob.glob('data/*.jpg')
#     folder = ""
#     img_name = ""
#     imgs = load_images(imgs_path)

# El número de colores dependerá del número de tarjetas que tengamos.
# El código de cada color se define ejecutando color_code.py
LIGHT_COLORS = {
    'red': (138,153,0),
    'pink': (139,64,141),
    'blue': (102,108,139),  # subrayador
    'aquamarine': (40,92,110),  # estuche
    'purple': (100,123,57)
}

DARK_COLORS = {
    'red': (255,255,136),
    'pink': (255,121,255),
    'blue': (112,165,219),  # subrayador
    'aquamarine': (97,165,219),  # estuche
    'purple': (139,255,169)
}

CODE_HSV2BGR = 54
CODE_BGR2HSV = 40

def hsv_to_bgr(imgs):
    bgr_imgs = []
    for img in imgs:
        bgr_imgs.append(cv2.cvtColor(img, CODE_HSV2BGR))
    return bgr_imgs

def bgr_to_hsv(imgs):
    hsv_imgs = []
    for img in imgs:
        hsv_imgs.append(cv2.cvtColor(img, CODE_BGR2HSV))
    return hsv_imgs

def color_segmentation(img, color: str):
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, CODE_HSV2BGR)
    mask = cv2.inRange(hsv_img, LIGHT_COLORS[color], DARK_COLORS[color])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, CODE_HSV2BGR)
    show_image(segmented_bgr, 'Segmented image')
    return mask

def color_detected(mask, thresshold=200) -> bool:
    area_color = np.count_nonzero(mask)
    if area_color > thresshold:
        return True
    else:
        return False

def instert_password():
    with open("password.txt", "r") as passw:
        line = passw.readline()
        password = line.split(sep=",")
    # for color in password:
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    i = 0
    folder_path = "data/password"
    while i < len(password):
        color = password[i]
        frame = picam.capture_array()
        if i < 10:
            frame_name = f"colors_0{i}.jpg"
        else:
            frame_name = f"colors_{i}.jpg"
        save_images(frame, frame_name, folder_path)
        mask = color_segmentation(frame, color)
        if color_detected(mask):
            print(f"The color {color} was detected correctly. Show the next color card")
            i += 1
        else:
            print(f"The correct color hasn't been detected")
        time.sleep(3)