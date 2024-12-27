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
    'purple': (100,65,60),
    'dark_yellow': (0,180,130),
    'light_yellow': (15,100,120),
    'orange': (0,145,120),
    'pink': (120,0,100),
    'dark_blue': (90,125,90),
    'light_blue': (90,55,100),
    'dark_green': (30,120,0),
    'light_green': (30,80,110)
}

DARK_COLORS = {
    'purple': (150,110,210),
    'dark_yellow': (30,255,255),
    'light_yellow': (35,170,255),
    'orange': (20,255,255),
    'pink': (170,255,255),
    'dark_blue': (150,255,255),
    'light_blue': (110,105,255),
    'dark_green': (70,255,255),
    'light_green': (70,120,255)
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

def color_detected(mask, thresshold=20000) -> bool:
    area_color = np.count_nonzero(mask)
    if area_color > thresshold:
        return True
    else:
        return False

def get_password():
    try:
        with open("password.txt", "r", encoding="utf-8") as passw:
            line = passw.readline()
            password = line.split(sep=",")
            password[-1] = password[-1].strip("\n")
        return password  # Si password.txt estuviera vacío, el código no daría errores
    except FileNotFoundError:
        print("El archivo password.txt no se encuentra en la carpeta actual")
    
def prueba_insert_password():
    imgs_path = glob.glob("./data/color_segmentation/colors_*.jpg")
    imgs = load_images(imgs_path)
    password = get_password()
    i = 0
    j = 0
    folder_path = os.path.join("./data", "password_prueba")
    create_folder(folder_path)
    while i < len(password):
        color = password[i]
        frame = imgs[j]
        if i < 10:
            frame_name = f"colors_0{i}_{j}.jpg"
        else:
            frame_name = f"colors_{i}_{j}.jpg"
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
            j = 0
            if i < len(password):
                print("Show the next color card")
            else:
                print("Security system disconnected")
        else:
            print(f"The correct color hasn't been detected")
            j += 1
        time.sleep(1)

def instert_password(tiempo_espera: int = 90) -> bool:
    password = get_password()
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