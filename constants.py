# CALIBRATION
# Number of inner corners of the image
WIDTH = 12
HEIGHT = 8
# Constants needed for refine_corners
ZERO_ZONE = (-1, -1)
NUM_ITERATIONS = 30
ACCURACY = 0.01

DX = 30
DY = 30

# SECURITY SYSTEM
# El código de cada color se define ejecutando color_code.py
LIGHT_COLORS = {
    'purple': (110,70,130),
    'dark_yellow': (20,55,180),
    'light_yellow': (25,15,180),
    'orange': (0,90,170),
    'pink': (150,45,170),
    'dark_blue': (95,195,160),
    'light_blue': (95,65,170),
    'dark_green': (50,175,105),
    'light_green': (35,75,160)
}

DARK_COLORS = {
    'purple': (150,110,210),
    'dark_yellow': (25,255,255),
    'light_yellow': (35,150,205),
    'orange': (20,255,255),
    'pink': (180,80,195),
    'dark_blue': (105,255,185),
    'light_blue': (105,130,195),
    'dark_green': (70,230,155),
    'light_green': (50,120,170)
}

CODE_HSV2BGR = 54
CODE_BGR2HSV = 40

# TRACKER
DESK_COLORS = [(0, 125, 25), (20, 255, 255)]
NET_COLORS = [(0, 198, 105), (255, 255, 255)]
PINGPONG_BALL_COLORS = [(0, 78, 160), (88, 255, 255)]  # Orange ball
# PINGPONG_BALL_COLORS = [(60, 150, 69), (86, 218, 184)]  # Blue ball
POINTS2WIN = 5