import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "oyster_shell/"
BG_PATH = os.path.join(ROOT_DIR, "background.tif")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")

def preprocess_oyster(image_path, bg_path):
    # Load the oyster image and the background
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    else:
        print(image_path, "Path does not exist!")
        exit()
    bg = cv2.imread(bg_path)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    diff = cv2.absdiff(img_blur, bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
    # _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
    return img, mask

def get_oyster_metrics(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get the largest contour (assuming it's the oyster)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    compactness = (perimeter**2) / (4 * np.pi * area + 1e-5)
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1]
    # Ensure we don't divide by zero and handle orientation
    major = max(width, height)
    minor = min(width, height)
    elongation = major / (minor + 1e-5)
    
    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "elongation": elongation,
        "contour": cnt
    }


def visualize_oyster_analysis(img, mask, metrics):
    if metrics is None:
        print("No oyster detected in this image.")
        return

    # Create a copy so we don't modify the original
    vis_img = img.copy()
    cnt = metrics['contour']

    # 1. Draw the actual contour (Green line, thickness 3)
    cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 3)

    # 2. Draw the Rotated Rect (Blue line) to visualize Elongation
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis_img, [box], 0, (255, 0, 0), 2)

    # Display the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask (After Morphology)")
    plt.imshow(mask, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Contour Analysis\nCompactness: {metrics['compactness']:.2f}\nElongation: {metrics['elongation']:.2f}")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig("plt.png")


if __name__ == "__main__":
    example_img = os.path.join(TRAIN_DIR, "Broken/251.tif")
    original, mask = preprocess_oyster(example_img, BG_PATH)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("og.png", original)

    metrics = get_oyster_metrics(mask)
    visualize_oyster_analysis(original, mask, metrics)

