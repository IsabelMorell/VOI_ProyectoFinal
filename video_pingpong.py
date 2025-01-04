import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List
import security_system as ss
import copy
import constants as cte

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
    time_margin = 30

    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    fps = calculate_fps(picam)  # Frame rate of the video

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
    output_folder_path = "./output"
    create_folder(output_folder_path)
    output_path = os.path.join(output_folder_path, "output_video_bounceNotDetected5.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    t_auxiliar = time.time()
    while (time.time() - t_auxiliar) <= time_margin:
        frame = picam.capture_array()
        out.write(frame)
    out.release()  
    cv2.destroyAllWindows()