import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List

if __name__ == "__main__":
    frame_width = 1280
    frame_height = 720
    fps = 110  # Frame rate of the video  https://picamera.readthedocs.io/en/release-1.13/api_camera.html
    frame_size = (frame_width, frame_height) # Size of the frames
    
    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
    
    output_folder_path = "./data"
    create_folder(output_folder_path)
    output_path = os.path.join(output_folder_path, "video_prueba.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    t_start = time.time()


    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()