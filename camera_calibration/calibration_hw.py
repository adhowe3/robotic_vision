import cv2 as cv
import os
import numpy as np

INPUT_IMG_FOLDER = "./calibration_images"
OUTPUT_IMG_FOLDER = "./output_images"


pattern_size = (10, 7)  # inner corners (columns, rows)
square_size = 1.0
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

def task_1(file_name):
    infile = os.path.join(INPUT_IMG_FOLDER, file_name)
    outfile = os.path.join(OUTPUT_IMG_FOLDER, file_name)
    os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)
    image = cv.imread(infile)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # Find chessboard corners
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
        return corners_refined
    else:
        print("Chessboard corners not found!")



def task_2():
    object_points = []
    image_points = []

    # get image size just use first image
    first_image = cv.imread( os.path.join(INPUT_IMG_FOLDER, "AR1.jpg"))
    h, w, c = first_image.shape
    image_size = (w, h)
    print(image_size)

    # Loop through all files in the folder
    for file_name in os.listdir(INPUT_IMG_FOLDER):
        if not file_name.lower().endswith(".jpg"):
            continue
        corners = task_1(file_name=file_name)
        if corners is not None:
            image_points.append(corners)
            object_points.append(objp)
        
    ret, camera_matrix, dist, revecs, tvecs = cv.calibrateCamera(objectPoints=object_points, imagePoints=image_points, imageSize=image_size, cameraMatrix=None, distCoeffs=None)
    
    p_size_mm = 0.0074
    if ret:
        print("camera_matrix:", camera_matrix)
        fx_p = camera_matrix[0,0]
        fx_mm = fx_p * p_size_mm
        fy_p = camera_matrix[1,1]
        fy_mm = fy_p * p_size_mm
        print("fx_p:", fx_p, "fy_p:", fy_p)
        print("fx_mm:", fx_mm)
        print("fy_mm:", fy_mm)
        print("")
        print("dist:", dist)


if __name__ == "__main__":

    task_2()
