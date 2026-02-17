import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "oyster_shell/"
BG_PATH = os.path.join(ROOT_DIR, "background.tif")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")

def preprocess_oyster(image_path, bg_path):
    # Load the oyster image and the background
    img = cv2.imread(image_path)
    bg = cv2.imread(bg_path)
    
    # Absolute difference between image and background
    # This highlights anything that ISN'T the background
    diff = cv2.absdiff(img, bg)
    
    # Convert to grayscale for thresholding
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Thresholding: pixels that are different enough from BG become white (255)
    _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    return img, mask

if __name__ == "__main__":
    example_img = os.path.join(TRAIN_DIR, "good/1.tif")
    original, mask = preprocess_oyster(example_img, BG_PATH)