import cv2 as cv
import os

INPUT_IMG_FOLDER = "./calibration_images"
OUTPUT_IMG_FOLDER = INPUT_IMG_FOLDER + "/output_images"

import cv2 as cv
import os

def task_1():
    infile = os.path.join(INPUT_IMG_FOLDER, "AR1.jpg")
    outfile = os.path.join(OUTPUT_IMG_FOLDER, "AR1.jpg")
    os.mkdir(OUTPUT_IMG_FOLDER)
    image = cv.imread(infile)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # Find chessboard corners
    pattern_size = (10, 7)  # inner corners (columns, rows)
    ret, corners = cv.findChessboardCorners(gray_image, pattern_size)
    
    if ret:
        # Refine corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        corners_refined = cv.cornerSubPix(gray_image, corners, win_size, zero_zone, criteria)
        
        # Draw corners on a color version of the grayscale image
        gray_color = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(gray_color, pattern_size, corners_refined, ret)
        
        # Show the image
        cv.imwrite(outfile, gray_color)
    else:
        print("Chessboard corners not found!")



def task_2():
    # Loop through all files in the folder
    for filename in os.listdir(INPUT_IMG_FOLDER):
        if not filename.lower().endswith(".jpg"):
            continue  # skip non-image files

        infile = os.path.join(INPUT_IMG_FOLDER, filename)
        image = cv.imread(infile)





if __name__ == "__main__":

    task_1()
