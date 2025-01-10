import os
import cv2
import glob
import time
import numpy as np
import constants as cte

from picamera2 import Picamera2
from typing import List,Tuple
from utils import *


def color_segmentation(img: np.ndarray, color: str) -> Tuple[np.ndarray,np.ndarray]:
    """
    Does the color segmentation of an image

    Args:
        img (np.ndarray): original image
        color (str): color we want to extract from the image

    Returns:
        np.ndarray: binary mask that extracts the color from the image
        np.ndarray: segmented image (img + mask)
    """
    hsv_img = cv2.cvtColor(img, cte.CODE_BGR2HSV)
    mask = cv2.inRange(hsv_img, cte.LIGHT_COLORS[color], cte.DARK_COLORS[color])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, cte.CODE_HSV2BGR)
    return mask, segmented_bgr

def color_detected(mask: np.ndarray, thresshold: int = 15000) -> bool:
    """
    Evaluates if the color has been detected 

    Args:
        mask (np.ndarray): binary mask that extracts the color from the image
        thresshold (int, optional): minimum number of pixels we want to have detected. 
            Defaults to 15000.

    Returns:
        bool: if the color has been detected or not
    """
    area_color = np.count_nonzero(mask)
    if area_color > thresshold:
        return True
    else:
        return False

def get_password(filepath: str) -> List[str]:
    """
    Reads the password from the secret file

    Args:
        filepath (str): path of the secret file

    Returns:
        List[str]: list of the colors of the password
    """
    try:
        with open(filepath, "r", encoding="utf-8") as passw:
            line = passw.readline()
            password = line.split(sep=",")
            password[-1] = password[-1].strip("\n")
        return password  # If pasword.txt was empty, the code won't crash
    except FileNotFoundError:
        print(f"The file {filepath} has not been found")
    
def prueba_insert_password() -> None:
    """
    Auxiliar function to test the process in a local environment, using images instead of 
    the camera of the Raspberrie Pi and real time frames
    """    
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
        mask, segmented_bgr = color_segmentation(frame, color)  # We obtain the mask of the color we wanted to identify

        # This is for saving the images for the report, inncecessary for the security systems correct functionality
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

def insert_password(picam: PiCamera2, out: cv2.VideoWriter, waitint_time: int = 90) -> bool:
    """
    Captures in real time each frame shown and analyzes if each color containdes in the password 
    is being read

    Args:
        picam (PiCamera2): camera object to capture the frame
        out (cv2.VideoWriter): object that register the frame that the camera captures
        waitint_time (int, optional): maximum time for entering the whole password. Defaults to 90.

    Returns:
        bool: if the password was correct and, therefore, the system was disconnected
    """
    password = get_password("password.txt")
    correct_password = False
    i = 0
    folder_path = "./data/password"
    create_folder(folder_path)
    init_time = time.time()
    while i < len(password):
        color = password[i]
        frame = picam.capture_array()  # Hacemos la foto
        # cv2.imshow("picam", frame)
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
            # cv2.imshow("picam", frame)
            out.write(frame) 

        if time.time() - init_time > waitint_time:
            break
    return correct_password, picam, out
    

if __name__=="__main__":
    prueba_insert_password()
