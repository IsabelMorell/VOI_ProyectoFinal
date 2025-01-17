from picamera2 import Picamera2
from typing import List, Tuple
from utils import *
import security_system as ss
import constants as cte
import time, cv2, copy
import numpy as np

def color_segmentation(img: np.ndarray, limit_colors: List[Tuple[int]]) -> Tuple[np.ndarray]:
    """
    Performs color-based segmentation on an input image using HSV color thresholds.

    Args:
        img (np.ndarray): input image in BGR format
        limit_colors (List[Tuple[int, int, int]]): HSV range for segmentation

    Returns:
        mask (np.ndarray): binary mask for segmented image
        segmented_bgr (np.ndarray): segmented image in BGR format
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, limit_colors[0], limit_colors[1])
    segmented = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
    return mask, segmented_bgr

def gaussian_blur(img: np.ndarray, sigma: float, filter_shape: List | None = None) ->  Tuple[np.ndarray]:
    """
    Performs a gaussian blur on a given image using a custom filter determined by sigma and filter_shape

    Args:
        img (np.ndarray): input image
        sigma (float): standard deviation (std) of the gaussian kernel.
        filter_shape (List | None): shape of the filter. If it's None, then it's computed based on sigma

    Returns:
        gaussian_filter (np.ndarray): filter applied
        gb_img.astype(np.uint8) (np.ndarray): blurred image
    """
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
    return gaussian_filter, gb_img.astype(np.uint8)

def sobel_edge_detector(img: np.ndarray, filter: np.ndarray, gauss_sigma: float, gauss_filter_shape: List | None = None) -> Tuple[np.ndarray]:
    """
    Detects edges on an image using the sobel kernel

    Args:
        img (np.ndarray): input image in BGR format
        filter (np.ndarray): filter to be applied
        gauss_sigma (float): standard deviation (std) of the gaussian kernel.
        gauss_filter_shape (List | None): shape of the gaussian filter. If it's None, then it's computed based on sigma

    Returns:
        np.squeeze(sobel_edges_img) (np.ndarray): image after the sobel edge detector was applied
        np.squeeze(theta) (np.ndarray): gradient angles at each pixel
    """
    # Transform the img to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get a blurry img to improve edge detections
    _, blurred = gaussian_blur(img=gray_img, sigma=gauss_sigma, filter_shape=gauss_filter_shape)
    
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
    return np.squeeze(sobel_edges_img), np.squeeze(theta)

def canny_edge_detector(img: np.ndarray, sobel_filter: np.ndarray, gauss_sigma: float, gauss_filter_shape: List | None = None) -> np.ndarray:
    """
    Implements the Canny edge detection algorithm

    Args:
        img (np.ndarray): input image in BGR format
        sobel_filter (np.ndarray): sobel filter to be applied
        gauss_sigma (float): standard deviation (std) of the gaussian kernel.
        gauss_filter_shape (List | None): shape of the gaussian filter. If it's None, then it's computed based on sigma

    Returns:
        canny_edges_img (np.ndarray): image with refined edges
    """
    # Call the method sobel_edge_detector()
    sobel_edges_img, theta = sobel_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape)
    
    # Use NMS to refine edges
    canny_edges_img = non_max_suppression(sobel_edges_img, theta)
    
    # Thresholding
    threshold = 0.5*canny_edges_img.max()
    canny_edges_img[canny_edges_img>threshold] = 255
    return canny_edges_img

def net_detection(frame: np.ndarray, net_colors: List[Tuple[int]], sobel_filter: np.ndarray, gauss_sigma: float, gauss_filter_shape: List | None = None) -> Tuple[int]:
    """
    Detects the edges of the net in a pingpong table as well as the location of the desk top. This is because it is 
    the same as the lower part of the net.
    For this, it will first apply a color-based segmentation on the input image, to locate the net. Then it will calculate
    the edges of the net using the canny edge algorithm and lastly it will select the pixel more on the left, the right and the
    bottom

    Args:
        frame (np.ndarray): input frame in BGR format
        net_colors (List[Tuple[int, int, int]]): HSV range for segmentation
        sobel_filter (np.ndarray): sobel filter to be applied
        gauss_sigma (float): standard deviation (std) of the gaussian kernel.
        gauss_filter_shape (List | None): shape of the gaussian filter. If it's None, then it's computed based on sigma

    Returns:
        left_net (int): x location of the left-side of the net
        right_net (int): x location of the right-side of the net
        desk_top (int): y location of the top-side of the desk
    """
    mask, segmented_net = color_segmentation(frame, net_colors)
    net_edges = canny_edge_detector(segmented_net, sobel_filter, gauss_sigma, gauss_filter_shape)     
    
    coords = np.column_stack(np.where(net_edges > 250))
    if coords.size > 0:
        left_net = np.min(coords[:, 1])
        right_net = np.max(coords[:, 1])
        desk_top = np.max(coords[:, 0])
        return left_net, right_net, desk_top

def desk_detection(frame: np.ndarray, desk_colors: List[Tuple[int]], sobel_filter: np.ndarray, gauss_sigma: float, gauss_filter_shape: List | None = None) -> Tuple[int]:
    """
    Detects the edges of the table.
    For this, it will first apply a color-based segmentation on the input image, to locate the desk. Then it will calculate
    its edges using the canny edge algorithm and lastly it will select the pixel more on the left and the right 

    Args:
        frame (np.ndarray): input frame in BGR format
        desk_colors (List[Tuple[int, int, int]]): HSV range for segmentation
        sobel_filter (np.ndarray): sobel filter to be applied
        gauss_sigma (float): standard deviation (std) of the gaussian kernel.
        gauss_filter_shape (List | None): shape of the gaussian filter. If it's None, then it's computed based on sigma

    Returns:
        left_limit (int): x location of the left-side of the desk
        right_limit (int): x location of the right-side of the desk
    """
    mask, segmented_desk = color_segmentation(frame, desk_colors)
    desk_edges = canny_edge_detector(segmented_desk, sobel_filter, gauss_sigma, gauss_filter_shape) 
    # We get the coordinates of the white pixels
    coords = np.column_stack(np.where(desk_edges > 250))

    if coords.size > 0:
        left_limit = np.min(coords[:, 1])
        right_limit = np.max(coords[:, 1])
        return left_limit, right_limit

def calculate_fps(picam) -> float:
    """
    Calculates the frames per second (FPS) of the picamera

    Args:
        picam: instance of the Picamera

    Returns:
        fps (float): frames per second
    """
    # Medir FPS
    num_frames = 120  # Número de frames para calcular FPS
    start_time = time.time()

    for _ in range(num_frames):
        frame = picam.capture_array()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    return fps

def check_bounce(x: float, y: float, x_prev: float, left_limit: int, left_net: int, right_net: int, right_limit: int, desk_top: int, saque: bool, num_bounces: int, score1: int, score2: int):
    """
    Given the location of a ball bounce, it determines its effect during gameplay

    Args:
        x (float): current x location of the ball
        y (float): current y location of the ball
        x_prev (float): previous x location of the ball
        left_limit, left_net, right_net, right_limit (int): table and net boundaries.
        desk_top (int): top boundary of the table
        saque (bool): indicates if the ball is being served
        num_bounces (int): number of ball bounces on the specific field
        score1 (int): score of player 1
        score2 (int): score of player 2
    
    Returns:
        Updated num_bounces (int), score1 (int), score2 (int), and end_point (bool)
    """
    end_point = False
    # Coordinate x of bounce to locate the field where it bounced
    if y > desk_top:  # it bounced on the floor
        if x_prev < x:  # Move made by P1
            if num_bounces == 0:
                score2 += 1
                end_point = True
            elif num_bounces == 1:
                if saque:
                    score2 += 1
                    end_point = True
                else:
                    score1 += 1
                    end_point = True
        else:  # Move made by P2
            if num_bounces == 0:
                score1 += 1
                end_point = True
            elif num_bounces == 1:
                if saque:
                    score1 += 1
                    end_point = True
                else:
                    score2 += 1
                    end_point = True
    elif x >= left_limit and x < left_net:
        if x_prev < x:  # Bounce in field of P1 made by P1
            if saque:
                if num_bounces == 0:
                    num_bounces += 1
                elif num_bounces == 1:
                    score2 += 1
                    end_point = True
            else:
                score2 += 1
                end_point = True
        else:  # Bounce in field of P1 move made by P2
            if saque and num_bounces == 0:
                score1 += 1
                end_point = True
            else:
                if num_bounces == 0:
                    num_bounces += 1
                elif num_bounces == 1:
                    score2 += 1
                    end_point = True
    elif x >= left_net and x <= right_net:  # it bounced on the net
        if x_prev < x:  # Move made by P1
            score2 += 1
            end_point = True
        else:  # Move made by P2
            score1 += 1
            end_point = True
    elif x > right_net and x < right_limit:
        if x_prev < x:  # Move made by P1
            if saque and num_bounces == 0:
                score2 += 1
                end_point = True
            else:
                if num_bounces == 0:
                    num_bounces += 1
                elif num_bounces == 1:
                    score1 += 1
                    end_point = True
        else:  # Move made by P2
            if saque:
                if num_bounces == 0:
                    num_bounces += 1
                elif num_bounces == 1:
                    score1 += 1
                    end_point = True
            else:
                score1 += 1
                end_point = True
        
    return num_bounces, score1, score2, end_point

def update_after_point() -> None:
    """
    Updates the parameters after a point is scored
    """
    global turn_player1, saque, num_bounces, x_prev, movement_prev
    turn_player1 = not turn_player1
    saque = True
    num_bounces = 0
    if turn_player1:
        x_prev = left_limit
        movement_prev = ["D", None]
    else:
        x_prev = right_limit
        movement_prev = ["I", None]

def check_winner(points2win: int, score1: int, score2: int) -> Tuple:
    """
    Checks if one of the players has won and who.

    Args:
        points2win (int): points needed to win a game
        score1 (int): points of player 1
        score2 (int): points of  player 2
    """
    if score1 >= points2win:
        return True, 1
    elif score2 >= points2win:
        return True, 2
    else:
        return False, None

def draw_score(frame: np.ndarray, frame_size: List, message: str, isScore: bool) -> np.ndarray:
    """
    Draws either the ame score or a message on the frame

    Args:
        frame (np.ndarray): input frame in BGR format
        frame_size (List): frame dimensions
        message (str): text to display
        isScore (bool): indicates if the message is a score or not

    Returns:
        frame (np.ndarray): input frame with the message included
    """
    frame_width, frame_height = frame_size
    if isScore:
        rect_width = 100
    else:
        rect_width = 500
    rect_height = 50
    rect_top_left = ((frame_width-rect_width)//2, frame_height-rect_height)
    rect_bottom_right = ((frame_width-rect_width)//2+rect_width, frame_height) 
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, color=(255, 255, 255), thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
    
    text_x = (frame_width - text_size[0]) // 2 
    text_y = frame_height - (rect_height // 2) + (text_size[1] // 2)
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    return frame

if __name__ == "__main__":
    frame_width = 1280
    frame_height = 720
    frame_size = (frame_width, frame_height) # Size of the frames
    time_margin = 10

    # Configuration to stream the video
    picam = Picamera2()
    picam.preview_configuration.main.size=frame_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    fps = calculate_fps(picam)  # Frame rate of the video

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec to use
    output_folder_path = "./output"
    create_folder(output_folder_path)
    output_path = os.path.join(output_folder_path, "output_tiempo_real_2jugadores.avi")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Security system
    correct_password, picam, out = ss.insert_password(picam, out)
    
    if correct_password:
        t_auxiliar = time.time()
        while (time.time() - t_auxiliar) <= time_margin:
            frame = picam.capture_array()
            out.write(frame)
        frame = picam.capture_array()

        # Parameters for the net detection
        sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gauss_sigma = 3
        gauss_filter_shape = [8*gauss_sigma + 1, 8*gauss_sigma + 1]
        
        # Determination of the players' fields
        left_limit, right_limit = desk_detection(frame, cte.DESK_COLORS, sobel_filter, gauss_sigma, gauss_filter_shape)
        left_net, right_net, desk_top = net_detection(frame, cte.NET_COLORS, sobel_filter, gauss_sigma, gauss_filter_shape)

        frame_copy = cv2.cvtColor(copy.deepcopy(frame), cv2.COLOR_BGR2GRAY)
        frame_copy[:, left_limit] = 250
        frame_copy[:, right_limit] = 250
        frame_copy[:,left_net] = 250
        frame_copy[:,right_net] = 250
        frame_copy[desk_top,:] = 250
        save_images(frame_copy, "fields", "./fotos_memoria")

        # Parameters for the background subtraction
        history = 100
        varThreshold = 50
        detectShadows = False
        mog2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)

        # Instances needed to calculate the number of bounces
        turn_player1 = True  # By convenience, player 1, who is the one on the left, will start serving
        saque = True
        end_point = True
        win = False

        num_bounces = 0
        score1 = 0
        score2 = 0
        x_prev = left_limit
        y_prev = None
        movement = [None, None]
        movement_prev = ["D", None]

        message = "Let the game begin!"
        print(message)
        frame = draw_score(frame, frame_size, message, False)
        for i in range(int(fps)*time_margin):
            out.write(frame)

        x = frame_width//2
        while not win:
            if end_point:
                if turn_player1:
                    while x > left_limit:
                        frame = picam.capture_array()
                        frame_auxiliar = copy.deepcopy(frame)
                        ball_mask, segmented_ball = color_segmentation(frame_auxiliar, cte.PINGPONG_BALL_COLORS)
                        mask = mog2.apply(segmented_ball)  # This detects only the orange points which have move (the ball)

                        coords = np.column_stack(np.where(mask > 0))  # Pixels of the ball
                        if coords.size > 0:
                            x = np.mean(coords[:, 1])
                            y = np.mean(coords[:, 0])
                        # Save the frame
                        frame = draw_score(frame, frame_size, f"{score1} - {score2}", True)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        out.write(frame)
                else:
                    while x < right_limit:
                        frame = picam.capture_array()
                        frame_auxiliar = copy.deepcopy(frame)
                        ball_mask, segmented_ball = color_segmentation(frame_auxiliar, cte.PINGPONG_BALL_COLORS)
                        mask = mog2.apply(segmented_ball)  # This detects only the orange points which have move (the ball)

                        coords = np.column_stack(np.where(mask > 0))  # Pixels of the ball
                        if coords.size > 0:
                            x = np.mean(coords[:, 1])
                            y = np.mean(coords[:, 0])
                        # Save the frame
                        frame = draw_score(frame, frame_size, f"{score1} - {score2}", True)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        out.write(frame)
                x_prev = x
                y_prev = None
                end_point = False

            frame = picam.capture_array()

            frame_auxiliar = copy.deepcopy(frame)
            ball_mask, segmented_ball = color_segmentation(frame_auxiliar, cte.PINGPONG_BALL_COLORS)
            mask = mog2.apply(segmented_ball)  # This detects only the orange points which have move (the ball)

            coords = np.column_stack(np.where(mask > 0))  # Pixels of the ball
            if coords.size > 0:
                x = np.mean(coords[:, 1])
                y = np.mean(coords[:, 0])
                if y_prev is not None:
                    if y > y_prev:  # Bajando
                        movement[1] = "B"
                    elif y < y_prev:  # Subiendo 
                        movement[1] = "S"
                    else:  # Movimiento horizontal
                        movement[1] = "H" 
                if x_prev is not None:
                    if x > x_prev:  # Derecha
                        movement[0] = "D"
                    else:
                        movement[0] = "I"

                if saque and ((turn_player1 and x > left_net) or ((not turn_player1) and x < right_net)):
                    saque = False
                    if num_bounces == 0:
                        if turn_player1:
                            score2 += 1
                        else:
                            score1 += 1
                        end_point = True
                        update_after_point()
                    num_bounces = 0  # Reestablish num_bounces to 0 because the ball is going to the other field
                
                if not end_point:
                    # Localizar los botes y cuántos hay
                    if movement_prev[1] is not None:
                        if movement_prev[1] == "B" and movement[1] == "S":
                            num_bounces, score1, score2, end_point = check_bounce(x, y, x_prev, left_limit, left_net, right_net, right_limit, desk_top, saque, num_bounces, score1, score2)

                    movement_prev[1] = movement[1]
                
                if end_point:  # update scores
                    update_after_point()
                else:
                    if movement_prev[0] != movement[0]:  # The ball changes direction
                        if x >= (right_net-10) and x <= (right_net+20):  # Ball hit the net
                            score1 += 1
                            end_point = True
                            update_after_point()
                        elif x >= (left_net-20) and x <= (left_net+10):
                            score2 += 1
                            end_point = True
                            update_after_point()
                        else:  # Player hit the ball back
                            num_bounces = 0
                    movement_prev[0] = movement[0]
            else:  # The ball hasn't move
                if score1 != 0 or score2 != 0 and not saque:  # Game has started
                    if x_prev < left_limit:  # ball out of range
                        if num_bounces == 0:
                            score1 += 1
                        elif num_bounces == 1:
                            score2 += 1
                        end_point = True
                        update_after_point()
                    elif x_prev > right_limit:
                        if num_bounces == 0:
                            score2 += 1
                        elif num_bounces == 1:
                            score1 += 1
                        end_point = True
                        update_after_point()
            
            x_prev = x
            y_prev = y

            # Save the frame
            frame = draw_score(frame, frame_size, f"{score1} - {score2}", True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out.write(frame)

            # Check if there's a winner
            win, winner = check_winner(cte.POINTS2WIN, score1, score2)

        message = f"And the winner is P{winner} {score1} - {score2}!"
        
    else:
        frame = picam.capture_array()
        message = "Incorrect password"

    print(message)
    frame = draw_score(frame, frame_size, message, False)
    for i in range(int(fps)*time_margin):
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()