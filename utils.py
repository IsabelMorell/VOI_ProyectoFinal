import os
import cv2

def create_folder(folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

def save_images(img, img_name: str, folder_path: str = "."):
    if img_name[-4:] != ".jpg":
        img_name = f"{img_name}.jpg"
    img_path = os.path.join(folder_path, img_name)
    cv2.imwrite(img_path, img)
    