import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List
import security_system as ss
import copy
import constants as cte

def color_segmentation(img, limit_colors):
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, limit_colors[0], limit_colors[1])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
    return mask, segmented_bgr


def calculate_fps(picam):
    # Medir FPS
    num_frames = 120  # Número de frames para calcular FPS
    start_time = time.time()

    for _ in range(num_frames):
        frame = picam.capture_array()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    return fps

if __name__ == "__main__":
    frame_width = 1280
    frame_height = 720
    frame_size = (frame_width, frame_height) # Size of the frames
    time_margin = 5

    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # fps = calculate_fps(picam)  # Frame rate of the video

    # Create a VideoWriter object to save the video
    folder_path = "./auxiliar_naranja"
    create_folder(folder_path)

    print("Empieza YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    contador = 0
    t_auxiliar = time.time()
    while (time.time() - t_auxiliar) <= time_margin:
        frame = picam.capture_array()
        mask, segmented_img = color_segmentation(frame, cte.PINGPONG_BALL_COLORS)
        if contador < 10:
            save_images(frame, f"a_frame_og_0{contador}", folder_path)
            save_images(mask, f"mask_0{contador}", folder_path)
            save_images(segmented_img, f"segmented_img_0{contador}", folder_path)
        else:
            save_images(frame, f"a_frame_og_{contador}", folder_path)
            save_images(mask, f"mask_{contador}", folder_path)
            save_images(segmented_img, f"segmented_img_{contador}", folder_path)
        contador += 1
    cv2.destroyAllWindows()