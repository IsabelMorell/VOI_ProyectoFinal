import time
import cv2
from picamera2 import Picamera2
import numpy as np
from utils import *
from typing import List
import constants as cte


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
    filename = "./data/color_segmentation/desk_1.jpg"
    img = cv2.imread(filename)
    
    ball_mask, segmented_ball = color_segmentation(img, cte.PINGPONG_BALL_COLORS)
    show_image(ball_mask)
    show_image(segmented_ball)

    # Calcular el gradiente entre la mask y la mask anterior para saber si la pelota esta bajando o subiendo
    coords = np.column_stack(np.where(ball_mask > 0))  # Pixeles azules que se han movido
    if coords.size > 0:
        print("pelota encontrada")
    else:
        print("pelota no encontrada")
    