import os
import cv2
import imageio
import glob
import os
import time

import numpy as np

# from picamera2 import Picamera2
from typing import List
from utils import *


# def load_images(filenames):
#     return [cv2.imread(filename) for filename in filenames]

# def show_image(img) -> None:
#     cv2.imshow("Chessboard Images", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    


# imgs_path = glob.glob('data/*.jpg')
#     folder = ""
#     img_name = ""
#     imgs = load_images(imgs_path)

# El número de colores dependerá del número de tarjetas que tengamos.
# El código de cada color se define ejecutando color_code.py
LIGHT_COLORS = {
    'purple': (110,70,130),
    'dark_yellow': (20,55,180),
    'light_yellow': (25,15,180),
    'orange': (0,90,170),  # hasta aquí está bien
    'pink': (150,45,170),
    'dark_blue': (95,195,160),
    'light_blue': (95,65,170),
    'dark_green': (50,175,105),
    'light_green': (35,75,160)
}

DARK_COLORS = {
    'purple': (150,110,210),
    'dark_yellow': (25,255,255),
    'light_yellow': (35,150,205),
    'orange': (20,255,255),  # hasta aquí está bien
    'pink': (180,80,195),
    'dark_blue': (105,255,185),
    'light_blue': (105,130,195),
    'dark_green': (70,230,155),
    'light_green': (50,120,170)
}

CODE_HSV2BGR = 54
CODE_BGR2HSV = 40

# def hsv_to_bgr(imgs):
#     bgr_imgs = []
#     for img in imgs:
#         bgr_imgs.append(cv2.cvtColor(img, CODE_HSV2BGR))
#     return bgr_imgs

# def bgr_to_hsv(imgs):
#     hsv_imgs = []
#     for img in imgs:
#         hsv_imgs.append(cv2.cvtColor(img, CODE_BGR2HSV))
#     return hsv_imgs

def color_segmentation(img, color: str):
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, CODE_BGR2HSV)
    mask = cv2.inRange(hsv_img, LIGHT_COLORS[color], DARK_COLORS[color])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, CODE_HSV2BGR)
    show_image(segmented_bgr)
    return mask, segmented_bgr

def color_detected(mask, thresshold=15000) -> bool:
    area_color = np.count_nonzero(mask)
    if area_color > thresshold:
        return True
    else:
        return False

def get_password(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as passw:
            line = passw.readline()
            password = line.split(sep=",")
            password[-1] = password[-1].strip("\n")
        return password  # Si password.txt estuviera vacío, el código no daría errores
    except FileNotFoundError:
        print("El archivo password.txt no se encuentra en la carpeta actual")
    
def prueba_insert_password():
    imgs_path = glob.glob("./data/color_segmentation/all_colors*.jpg")
    imgs = load_images(imgs_path)
    password = get_password("password_prueba.txt")
    i = 0
    j = -1
    folder_path = os.path.join("./data", "password_prueba")
    create_folder(folder_path)
    while i < len(password):
        color = password[i]
        frame = imgs[-1]
        if i < 10:
            frame_name = f"colors_0{i}.jpg"
        else:
            frame_name = f"colors_{i}.jpg"
        # show_image(frame)
        mask, segmented_bgr = color_segmentation(frame, color)  # Sacamos la máscara del color que deberíamos identificar

        # Esto es para guardar imágenes
        # cv2.imshow("Color Segmented Image", segmented_bgr)
        # key = cv2.waitKey(0)
        # # If the key is 's' select the frame as the reference frame
        # if key == ord('s'):
        #     save_images(segmented_bgr, frame_name, folder_path)

        if color_detected(mask):
            print(f"The color {color} was detected correctly")
            save_images(segmented_bgr, frame_name, folder_path)
            i += 1
            if i < len(password):
                print("Show the next color card")
            else:
                print("Security system disconnected")
        else:
            print(f"The correct color hasn't been detected")
        time.sleep(1)

def instert_password(tiempo_espera: int = 90) -> bool:
    password = get_password("password.txt")
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    correct_password = False
    i = 0
    folder_path = "./data/password"
    create_folder(folder_path)
    tiempo_inicio = time.time()
    while i < len(password):
        color = password[i]
        frame = picam.capture_array()  # Hacemos la foto
        if i < 10:
            frame_name = f"colors_0{i}.jpg"
        else:
            frame_name = f"colors_{i}.jpg"
        mask, segmented_bgr = color_segmentation(frame, color)  # Sacamos la máscara del color que deberíamos identificar
        if color_detected(mask):
            print(f"The color {color} was detected correctly")
            save_images(segmented_bgr, frame_name, folder_path)
            i += 1
            if i < len(password):
                print("Show the next color card")
            else:
                print("Security system disconnected")
                correct_password = True
        else:
            print(f"The correct color hasn't been detected")
        time.sleep(1.5)

        if time.time() - tiempo_inicio > tiempo_espera:
            break
    return correct_password
    

if __name__=="__main__":
    prueba_insert_password()