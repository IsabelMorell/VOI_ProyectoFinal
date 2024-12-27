import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List

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
    
    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    fps = calculate_fps(picam)

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


videoname = f'output_{history}_{varThreshold}_{detectShadows}.avi' # Name of the output video file with the parameters
    mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
    frame_size = (frame_width, frame_height) # Size of the frames
    fps = frame_rate # Frame rate of the video
    path = os.path.join(folder_path, videoname)
    out = cv2.VideoWriter(path, fourcc, fps, frame_size)

    for frame in frames:
        # Apply the MOG2 algorithm to detect the moving objects
        mask = mog2.apply(frame)
        # Convert to BGR the mask to store it in the video
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Save the mask in a video
        out.write(mask)

    out.release()
