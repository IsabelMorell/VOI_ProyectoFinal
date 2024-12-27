import time
import cv2
from picamera2 import Picamera2
import constants as cte
import numpy as np
from utils import *
from typing import List

def color_segmentation(img, color):
    # Necesitamos saber cómo viene la imagen para saber si hay que pasarla a hsv o no. Asumo que vienen en BGR
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, LIGHT_COLORS[color], DARK_COLORS[color])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, CODE_HSV2BGR)
    show_image(segmented_bgr)
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
    
    if verbose:
        show_image(canny_edges_img, img_name="Canny Edges")   
    return canny_edges_img


def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel:float, minDistance: int, corner_color: tuple, radius: int):
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
    filename = "./data/foto_auxiliar.jpeg"
    img = cv2.imread(filename)

    color_segmentation(img, color)

    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gauss_sigma = 3
    gauss_filter_shape = [8*gauss_sigma + 1, 8*gauss_sigma + 1]

    canny_edge = canny_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape) 

    show_image(canny_edge, "Canny edges")

    desk_corners = desk_corners_detection(img)

    show_image(desk_corners, "esquinas_mesa")