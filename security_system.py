import os
import cv2
import imageio
import glob
import os

import numpy as np

from picamera2 import Picamera2
from typing import List
from utils_lab2 import non_max_suppression, get_hsv_color_ranges


# imgs_path = glob.glob('data/*.jpg')
#     folder = ""
#     img_name = ""
#     imgs = load_images(imgs_path)

# El número de colores dependerá del número de tarjetas que tengamos.
# El código de cada color se define ejecutando color_code.py
LIGHT_COLORS = {
    'red': (0,0,0),
    'orange': (0,0,0),
    'yellow': (0,0,0),
    'green': (0,0,0),
    'blue': (0,0,0),
    'blue': (0,0,0),
}

DARK_COLORS = {
    'red': (0,0,0),
    'orange': (0,0,0),
    'yellow': (0,0,0),
    'green': (0,0,0),
    'blue': (0,0,0),
    'blue': (0,0,0),
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
    while i < len(password):
        color = password[i]
        frame = picam.capture_array()
        mask = color_segmentation(frame, color)
        if color_detected(mask):
            print(f"The color {color} was detected correctly. Show the next color card")
            i += 1
        else:
            print(f"The correct color hasn't been detected")