"""
ahora mismo estoy haciendo el background substraction en tiempo real y guardando el video de la partida a la vez
necesito averiguar el fps de la camara
"""
import cv2
from picamera2 import Picamera2
import constants as cte
import numpy as np
from utils import *
from typing import List

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
    
    if verbose:
        show_image(canny_edges_img, img_name="Canny Edges")   
    return canny_edges_img

def net_detection(frame: np.array, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None):
    frame_segmented = color_segmentation(frame, cte.NET_COLOR)
    # Si se detectan varios objetos a parte de la red habrá que aplicar algún operador
    # morfológico como erosión
    canny_edge = canny_edge_detector(frame_segmented, sobel_filter, gauss_sigma, gauss_filter_shape) 
    return canny_edge

# Define Shi-Tomasi corner detection function
def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel:float, minDistance: int, corner_color: tuple, radius: int):
    '''
    image - Input image
    maxCorners - Maximum number of corners to return. 
                 If there are more corners than are found, the strongest of them is returned. 
                 maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned
    qualityLevel - Parameter characterizing the minimal accepted quality of image corners. 
                   The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. 
                   The corners with the quality measure less than the product are rejected. 
                   For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected
    minDistance - Minimum possible Euclidean distance between the returned corners
    corner_color - Desired color to highlight corners in the original image
    radius - Desired radius (pixels) of the circle
    '''
    # Input image to Tomasi corner detector should be grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply cv2.goodFeaturesToTrack function
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    # Corner coordinates conversion to integers
    corners = np.intp(corners)
    for corner in corners:
        # Multidimensional array into flattened array, if necessary
        x, y = corner.ravel()
        # Circle corners
        cv2.circle(image, (x,y), radius, corner_color)
    return image

def desk_corners_detection(image: np.array, maxCorners = 100, qualityLevel = 0.1, minDistance = 4,  corner_color = (255, 0, 255), radius = 4):
    corners = shi_tomasi_corner_detection(image, maxCorners, qualityLevel, minDistance, corner_color, radius)
    return corners

if __name__ == "__main__":
    frame_width = 1280
    frame_height = 720
    fps =  # Frame rate of the video
    frame_size = (frame_width, frame_height) # Size of the frames

    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    frame = picam.capture_array()

    # Parameters for the net detection
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gauss_sigma = 3
    gauss_filter_shape = [8*gauss_sigma + 1, 8*gauss_sigma + 1]
    img_net = net_detection(frame, sobel_filter, gauss_sigma, gauss_filter_shape)
    desk_corners = desk_corners_detection(frame)

    # Parameters for the background subtraction
    history = 100
    varThreshold = 50
    detectShadows = False
    mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
    
    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
    
    output_path = os.path.join("data", "output_video.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Comienzo la sustraccion de fondo en tiempo real
        
    cv2.destroyAllWindows()