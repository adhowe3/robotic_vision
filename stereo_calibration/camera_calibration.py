import cv2 as cv
import os
import numpy as np

INPUT_IMG_FOLDER = "./input_images"
OUTPUT_IMG_FOLDER = "./output_images"
LEFT_IMG_FOLDER = "./braden_image_calibration/L"
RIGHT_IMG_FOLDER = "./braden_image_calibration/R"
STEREO_R_IMG_FOLDER = "./braden_image_calibration/SR"
STEREO_L_IMG_FOLDER = "./braden_image_calibration/SL"

PRACTICE_L = "practice_img/L"
PRACTICE_R = "practice_img/R"

pattern_size = (10, 7)  # inner corners (columns, rows) UNCOMMENT FOR TASK_2()
square_size = 3.985  # this is in inches (the EB large board)
# square_size = 2  # this is in inches (The practice images)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

# function for taking in file and finding chess board corners
def task_0(file_name, input_path, output_path):
    infile = os.path.join(input_path, file_name)
    outfile = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)
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
        print("Chessboard corners not found! - ", infile)

# function task_2 from previous hw, gets the camera intrensics
def task_1(input_path, output_path):
    object_points = []
    image_points = []
    file_type = ".bmp"
    # file_type = ".png"

    # get image size just use first image
    for file_name in os.listdir(input_path):
        if not file_name.lower().endswith(file_type):
            continue
        first_image = cv.imread(os.path.join(input_path, file_name))
        h, w, c = first_image.shape
        image_size = (w, h)
        print(image_size)
        break

    # Loop through all files in the folder
    for file_name in os.listdir(input_path):
        if not file_name.lower().endswith(file_type):
            continue
        corners = task_0(file_name, input_path, output_path)
        if corners is not None:
            image_points.append(corners)
            object_points.append(objp)
        
    ret, camera_matrix, dist, revecs, tvecs = cv.calibrateCamera(objectPoints=object_points, imagePoints=image_points, imageSize=image_size, cameraMatrix=None, distCoeffs=None)
    
    if ret:
        print("camera_matrix:", camera_matrix)
        fx_p = camera_matrix[0,0]
        fy_p = camera_matrix[1,1]
        print("fx_p:", fx_p, "fy_p:", fy_p)
        print("dist:", dist)
        np.savez("camera_parameters_test_test_left.npz", camera_matrix=camera_matrix, dist=dist)


if __name__ == "__main__":
    task_1(input_path=PRACTICE_L, output_path="output_path")
