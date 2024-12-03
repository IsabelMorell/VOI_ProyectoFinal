import cv2
from picamera2 import Picamera2
import os
import time
from utils import *

def stream_video(folder_path, threshold, duration):
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
                frame_name = f"chessboard_0{contador}.jpg"
            else:
                frame_name = f"chessboard_{contador}.jpg"
            cv2.imshow("picam", frame)
            save_images(frame, frame_name, folder_path)
            contador += 1
            tiempo_ant = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threshold = 1
    duration = 30
    folder_path = os.path.join("./data", "chessboard_frames")
    create_folder(folder_path)
    stream_video(folder_path, threshold, duration)