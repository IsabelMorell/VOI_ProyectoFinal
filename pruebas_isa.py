import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List

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


if __name__ == "__main__":
    """ DETERMINACION DE LOS CAMPOS DE LOS JUGADORES """
    filename = "./data/color_segmentation/desk_0.jpg"
    img = cv2.imread(filename)
    DESK_COLORS = [(0, 125, 25), (20, 255, 255)]

    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gauss_sigma = 3
    gauss_filter_shape = [8*gauss_sigma + 1, 8*gauss_sigma + 1]

    """mask, segmented_img = color_segmentation(img, DESK_COLORS)
    show_image(mask, "Mask")
    show_image(segmented_img, "Segmented image")

    desk_edges = canny_edge_detector(segmented_img, sobel_filter, gauss_sigma, gauss_filter_shape) 

    show_image(desk_edges, "Desk edges")
    # Calcula las coordenadas de los píxeles blancos
    coords = np.column_stack(np.where(desk_edges > 250))

    if coords.size > 0:  # Verifica si hay píxeles que cumplan la condición
        # Encuentra las coordenadas X de los píxeles más a la izquierda y a la derecha
        left_limit = np.min(coords[:, 1])  # Mínima coordenada X
        right_limit = np.max(coords[:, 1])
    
        desk_edges[:,left_limit] = 250
        desk_edges[:,right_limit] = 250

        show_image(desk_edges, "Desk edges")
        save_images(desk_edges, "desk_edges", "./fotos_memoria")
    
    NET_COLOR = [(0, 198, 105), (255, 255, 255)]
    mask, segmented_net = color_segmentation(img, NET_COLOR)
    #show_image(mask, "Mask")
    #show_image(segmented_net, "Segmented net")

    net_edges = canny_edge_detector(segmented_net, sobel_filter, gauss_sigma, gauss_filter_shape) 

    #show_image(net_edges, "net edges")
    coords = np.column_stack(np.where(net_edges > 250))

    if coords.size > 0:  # Verifica si hay píxeles que cumplan la condición
        # Encuentra las coordenadas X de los píxeles más a la izquierda y a la derecha
        left_net = np.min(coords[:, 1])  # Mínima coordenada X
        right_net = np.max(coords[:, 1])

        desk_top = np.max(coords[:, 0])

        net_edges[:,left_net] = 250
        net_edges[:,right_net] = 250
        net_edges[desk_top,:] = 250

        show_image(net_edges, "Net edges")
        save_images(net_edges, "net_edges_desk_top", "./fotos_memoria")
    """
    
    # Use VideoCapture from OpenCV to read the video
    videopath = './data/video_prueba.avi'  # Path to the video file
    cap = cv2.VideoCapture(videopath)  

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

    PINGPONG_BALL_COLORS = [(0, 172, 130), (166, 255, 255)]

    # Parameters for the background subtraction
    history = 100
    varThreshold = 50
    detectShadows = False
    mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
    
    # Intancias previas al bucle
    num_botes = 0
    y_prev = None
    position_prev = None
    print("num frames", len(frames))

    for frame in frames:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        """ball_mask, segmented_ball = color_segmentation(frame, PINGPONG_BALL_COLORS)
        
        # Comienzo la sustraccion de fondo en tiempo real
        mask = mog2.apply(segmented_ball)  # Esto es lo que se ha movido (osea la pelota)

        coords = np.column_stack(np.where(mask > 0))  # Encuentra los píxeles blancos
        if coords.size > 0:
            y = np.mean(coords[:, 0])  # Promedio de las coordenadas Y
            if y_prev is not None:
                if y > y_prev:  # Bajando
                    position = "B"
                elif y < y_prev:  # Subiendo 
                    position = "S"
                else:  # No hay movimiento vertical
                    position = "H"  # Horizontal
                
                if position_prev is not None and position == "S" and position_prev == "B":
                    num_botes += 1
                    print(num_botes)
                position_prev = position
            y_prev = y"""
    
    print("numero de botes total =", num_botes)