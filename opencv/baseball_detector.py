import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import os

IN_IMG_PATH = "./baseball_images/"
OUT_IMG_PATH = "./baseball_images_out/"

def hough_circle(gray):
    # gray_blur = cv.GaussianBlur(gray, (7,7), 0)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1,              # accumulator resolution
        minDist=30,        # min distance between circles
        param1=100,        # higher threshold for Canny edge
        param2=30,         # accumulator threshold for circle detection
        minRadius=3,      # min radius of ball
        maxRadius=50       # max radius of ball
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            # cv.circle(gray, (x, y), r, (255, 0, 0), 2)  # draw circle
            cv.circle(gray, (x, y), 2, (0,0,255), 3)    # draw center
    return gray


prev_gray = None

def differencing(image):
    global prev_gray
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray.copy()
        return np.zeros_like(gray)
    diff = cv.absdiff(gray, prev_gray)
    prev_gray = gray.copy()
    return diff

def contours(diff, original, min_area=50):
    _, mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    mask = cv.GaussianBlur(mask, (5,5), 0)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours_list, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw directly on the original image
    output = original.copy()

    for cnt in contours_list:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        # cv.circle(output, center, radius, (0, 255, 255), 2)
        cv.circle(output, center, 2, (0, 255, 255), -1)

    return output

def detect_baseball(image):
    diff = differencing(image)
    cont = contours(diff, image)
    return cont



if __name__ == "__main__":

    # Loop through all files in the folder
    for filename in os.listdir(IN_IMG_PATH):
        if not filename.lower().endswith(".png"):
            continue  # skip non-image files

        infile = os.path.join(IN_IMG_PATH, filename)
        image = cv.imread(infile)
        if image is None:
            print(f"Could not read {filename}, skipping")
            continue

        result = detect_baseball(image)  # main detect function

        # Save output
        outfile = os.path.join(OUT_IMG_PATH, filename)
        cv.imwrite(outfile, result)
        print(f"Processed and saved {filename}")

