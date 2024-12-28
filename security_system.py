import os
import cv2
import glob
import os
import time

import numpy as np
import constants as cte

# from picamera2 import Picamera2
from typing import List,Tuple
from utils import *


def color_segmentation(img: np.ndarray, color: str) -> Tuple[np.ndarray,np.ndarray]:
    # DOCUMENTACIÓN COPIADA DE CHAT => Revisar y añadir el error 
    """
    Realiza la segmentación de color en una imagen dada y devuelve una máscara binaria y la imagen segmentada.
    
    Args:
        img (np.array): Imagen de entrada en formato BGR (matriz de NumPy).
        color (str): Color objetivo a segmentar (por ejemplo, 'red', 'blue', 'green').
    
    Returns:
        Tuple[np.array, np.array]: 
            - La máscara binaria (1 donde se detecta el color, 0 en el resto).
            - La imagen segmentada en formato BGR.
    
    Raises:
        ValueError: Si el color proporcionado no es válido.
    """
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, cte.CODE_BGR2HSV)
    mask = cv2.inRange(hsv_img, cte.LIGHT_COLORS[color], cte.DARK_COLORS[color])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, cte.CODE_HSV2BGR)
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
    folder_path = os.path.join("./data", "password_prueba")
    create_folder(folder_path)
    while i < len(password):
        color = password[i]
        frame = imgs[-1]
        if i < 10:
            frame_name = f"colors_0{i}.jpg"
        else:
            frame_name = f"colors_{i}.jpg"
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

def insert_password(picam, out, tiempo_espera: int = 90) -> bool:
    password = get_password("password.txt")
    correct_password = False
    i = 0
    folder_path = "./data/password"
    create_folder(folder_path)
    tiempo_inicio = time.time()
    while i < len(password):
        color = password[i]
        frame = picam.capture_array()  # Hacemos la foto
        cv2.imshow("picam", frame)
        out.write(frame) 

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
        
        t_auxiliar = time.time()
        while (time.time() - t_auxiliar) < 1.5:  # time.sleep(1.5)
            frame = picam.capture_array()  # Hacemos la foto
            cv2.imshow("picam", frame)
            out.write(frame) 

        if time.time() - tiempo_inicio > tiempo_espera:
            break
    return correct_password, picam, out
    

if __name__=="__main__":
    prueba_insert_password()