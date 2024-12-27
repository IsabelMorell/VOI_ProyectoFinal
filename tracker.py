"""
ahora mismo estoy haciendo el background substraction en tiempo real y guardando el video de la partida a la vez
necesito averiguar el fps de la camara
"""
import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List
import security_system as ss

def color_segmentation(img, limit_colors):
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, limit_colors[0], limit_colors[1])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
    return mask, segmented_bgr

def gaussian_blur(img: np.array, sigma: float, filter_shape: List | None = None, verbose: bool = False) -> np.array:
    # If not given, compute the filter shape 
    if filter_shape == None:
        filter_shape = [8*sigma + 1, 8*sigma + 1]
        filter_l = 8*sigma + 1
    else:
        filter_l = filter_shape[0]
    
    # Create the filter coordinates matrices
    y, x = np.mgrid[(-filter_l//2):filter_l//2, (-filter_l//2):filter_l//2]
    
    # Define the formula that goberns the filter
    formula = np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*sigma**2*np.pi)
    gaussian_filter = formula/formula.sum()
    
    # Process the image
    gb_img = cv2.filter2D(img, ddepth=-1, kernel=gaussian_filter)  
    
    if verbose:
        show_image(img=gb_img, img_name=f"Gaussian Blur: Sigma = {sigma}")    
    return gaussian_filter, gb_img.astype(np.uint8)

def sobel_edge_detector(img: np.array, filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False) -> np.array:
    # Transform the img to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get a blurry img to improve edge detections
    _, blurred = gaussian_blur(img=gray_img, sigma=gauss_sigma, filter_shape=gauss_filter_shape, verbose=verbose)
    
    # Re-scale
    blurred = blurred/255.0
    
    # Get vertical edges
    v_edges = cv2.filter2D(blurred, ddepth=-1, kernel=filter)
    
    # Transform the filter to get the orthogonal edges
    filter = np.transpose(filter)
    
    # Get horizontal edges
    h_edges = cv2.filter2D(blurred, ddepth=-1, kernel=filter)
    
    # Get edges
    sobel_edges_img = np.hypot(v_edges, h_edges)
    
    # Get edges angle
    theta = np.arctan2(h_edges, v_edges)
    
    # Visualize if needed
    if verbose:
        show_image(img=sobel_edges_img, img_name="Sobel Edges")    
    return np.squeeze(sobel_edges_img), np.squeeze(theta)

def canny_edge_detector(img: np.array, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False):
    # Call the method sobel_edge_detector()
    sobel_edges_img, theta = sobel_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape, verbose=verbose)
    
    # Use NMS to refine edges
    canny_edges_img = non_max_suppression(sobel_edges_img, theta)
    
    # Thresholding
    threshold = 0.5*canny_edges_img.max()
    canny_edges_img[canny_edges_img>threshold] = 255

    if verbose:
        show_image(canny_edges_img, img_name="Canny Edges")   
    return canny_edges_img

def net_detection(frame: np.array, net_colors: List, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None):
    mask, segmented_net = color_segmentation(frame, net_colors)
    net_edges = canny_edge_detector(segmented_net, sobel_filter, gauss_sigma, gauss_filter_shape)     
    
    coords = np.column_stack(np.where(net_edges > 250))
    if coords.size > 0:
        left_net = np.min(coords[:, 1])
        right_net = np.max(coords[:, 1])
        desk_top = np.max(coords[:, 0])
        return left_net, right_net, desk_top

def desk_detection(frame: np.array, desk_colors: List, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None):
    mask, segmented_desk = color_segmentation(frame, desk_colors)
    desk_edges = canny_edge_detector(segmented_desk, sobel_filter, gauss_sigma, gauss_filter_shape) 
    # We get the coordinates of the white pixels
    coords = np.column_stack(np.where(desk_edges > 250))

    if coords.size > 0:
        left_limit = np.min(coords[:, 1])
        right_limit = np.max(coords[:, 1])
        return left_limit, right_limit

if __name__ == "__main__":
    frame_width = 1280
    frame_height = 720
    fps = 30  # Frame rate of the video  https://picamera.readthedocs.io/en/release-1.13/api_camera.html
    frame_size = (frame_width, frame_height) # Size of the frames
    time_margin = 5
    DESK_COLORS = [(0, 125, 25), (20, 255, 255)]
    NET_COLORS = [(0, 198, 105), (255, 255, 255)]
    PINGPONG_BALL_COLORS = [(0, 172, 130), (166, 255, 255)]

    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Security system
    correct_password = ss.insert_password(picam)

    # TODO: guardamos en el video el sistema de seguridad, si no?

    if correct_password:
        time.sleep(time_margin)
        frame = picam.capture_array(picam)

        # Parameters for the net detection
        sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gauss_sigma = 3
        gauss_filter_shape = [8*gauss_sigma + 1, 8*gauss_sigma + 1]
        
        # Determination of the players' fields
        left_limit, right_limit = desk_detection(frame, DESK_COLORS, sobel_filter, gauss_sigma, gauss_filter_shape)
        left_net, right_net, desk_top = net_detection(frame, sobel_filter, NET_COLORS, gauss_sigma, gauss_filter_shape)

        # Parameters for the background subtraction
        history = 100
        varThreshold = 50
        detectShadows = False
        mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
        
        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
        
        output_folder_path = "./data/output"
        create_folder(output_folder_path)
        output_path = os.path.join(output_folder_path, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while True:
            frame = picam.capture_array()
            cv2.imshow("picam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ball_mask, segmented_ball = color_segmentation(frame, PINGPONG_BALL_COLORS)
            
            # Comienzo la sustraccion de fondo en tiempo real
            mask = mog2.apply(segmented_ball)  # Esto es lo que se ha movido (osea la pelota)

            # TODO: calcular el gradiente entre la mask y la mask anterior para saber si la pelota esta bajando o subiendo

            # Localizar los botes y cuántos hay

            # mostrar la puntuacion
            
            # Guardo el frame
            out.write()  # TODO: guardar el frame final que quiero que se grabe
            # a lo mejor quiere que se muestre en la camara del ordenador??

        out.release()
            
        cv2.destroyAllWindows()
    else:
        print("Incorrect password")
