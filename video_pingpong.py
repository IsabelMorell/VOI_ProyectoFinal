import cv2
import os
import numpy as np
import copy

if __name__ == "__main__":
    # Use VideoCapture from OpenCV to read the video
    videopath = './output/output_video_bounceNotDetected5.avi'  # Path to the video file
    cap = cv2.VideoCapture(videopath)  

    #Check if the video was successfully opened
    if not cap.isOpened():
        print('Error: Could not open the video file')
        os._exit(0)

    #Get the size of frames and the frame rate of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    #Use a loop to read the frames of the video and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    #Show the frames to select the reference frame, press 'n' to move to the next frame and 's' to select the frame
    for i, frame in enumerate(frames):
        # Show the frame
        cv2.imshow('Video', frame)

    cv2.destroyAllWindows()