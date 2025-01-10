import cv2
from picamera2 import Picamera2
import os
import time
from utils import *

def stream_video(folder_path: str, threshold: int, duration: int) -> None:
    """
    Starts a video that saves a frame (every time the threshold seconds have passed) 
    into the folder indicated for the amount of time received

    Args:
        folder_path (str): Path of the folder where we are saving the frames
        threshold (int): amount of time we wait between each take, in seconds
        duration (int): total duration of the process, in seconds
    """
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    contador = 0
    tiempo_ant = time.time()
    while contador <= duration:
        frame = picam.capture_array()
        
        if time.time() - tiempo_ant >= threshold:
            if contador < 10:
                frame_name = f"colors_0{contador}.jpg"
                # frame_name = f"all_colors_{contador}.jpg"
                # frame_name = f"desk_{contador}.jpg"
                # frame_name = f"pingpong_ball_{contador+2}.jpg"
            else:
                frame_name = f"colors_{contador}.jpg"
            cv2.imshow("picam", frame)
            save_images(frame, frame_name, folder_path)
            contador += 1
            tiempo_ant = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threshold = 1
    duration = 10
    folder_path = os.path.join("./data", "color_segmentation")
    create_folder(folder_path)
    stream_video(folder_path, threshold, duration)