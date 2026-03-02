import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

INPUT_IMG_FOLDER = "./input_images"
OUTPUT_IMG_FOLDER = "./output_images"
STEREO_R_IMG_FOLDER = "./braden_image_calibration/SR"
STEREO_L_IMG_FOLDER = "./braden_image_calibration/SL"

PRACTICE_SL = "practice_img/SL"
PRACTICE_SR = "practice_img/SR"

def load_camera_parameters(npz_file):
    data = np.load(npz_file)
    K = data["camera_matrix"]
    dist = data["dist"]
    return K, dist

def find_corners(image_path, pattern_size):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray,
        pattern_size,
        cv.CALIB_CB_ADAPTIVE_THRESH +
        cv.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        print("corners not found for ", image_path)
        return None
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners


def show_image(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def stereo_calibrate_from_files(
    left_folder,
    right_folder,
    pattern_size,
    square_size,
    CAMERA_PARAM_LEFT,
    CAMERA_PARAM_RIGHT
):
    # Load intrinsics
    K1, dist1 = load_camera_parameters(CAMERA_PARAM_LEFT)
    K2, dist2 = load_camera_parameters(CAMERA_PARAM_RIGHT)

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    object_points = []
    image_points_L = []
    image_points_R = []

    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    # loop through the images of left and right pairs
    for l_img, r_img in zip(left_images, right_images):
        if not l_img.lower().endswith(".png"):
            continue
        # find the corners for each
        corners_L = find_corners(os.path.join(left_folder, l_img), pattern_size)
        corners_R = find_corners(os.path.join(right_folder, r_img), pattern_size)

        if corners_L is None or corners_R is None:
            continue
        # append to each object points list
        object_points.append(objp)
        image_points_L.append(corners_L)
        image_points_R.append(corners_R)

    # Image size
    sample_img = cv.imread(os.path.join(left_folder, left_images[0]))
    h, w = sample_img.shape[:2]
    image_size = (w, h)

    flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

    # calculate the stereoCalibrate values
    rms, stereo_mtxL, stereo_distL, stereo_mtxR, stereo_distR, R, T, E, F = cv.stereoCalibrate(
        object_points,
        image_points_L,
        image_points_R,
        K1, dist1,
        K2, dist2,
        image_size,
        criteria=criteria,
        flags=flags
    )

    # save the stereo parameters
    np.savez(f"camera_params_stereo.npz", 
         stereo_mtxL=stereo_mtxL,
         stereo_distL=stereo_distL,
         stereo_mtxR=stereo_mtxR,
         stereo_distR=stereo_distR,
         R=R,
         T=T,
         E=E,
         F=F)

    rvec, _ = cv.Rodrigues(R)
    rvec_deg = np.degrees(rvec) # convert to degrees

    print("Stereo RMS error:\n", rms)
    print("\nRotation matrix (3x3):\n", R)
    print("\nTranslation vector T (3x1):\n", T)
    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)
    print("\nRotation vector (3x1) [radians]:\n", rvec)
    print("Rotation vector (3x1) in degrees:\n", rvec_deg)
    return R, rvec, T, E, F


def draw_points(img, points, color):
    for p in points:
        cv.circle(img, tuple(p.astype(int)), 6, color, -1)


def draw_epilines(img, lines, color):
    h, w = img.shape[:2]
    for line in lines:
        a, b, c = line
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
        cv.line(img, (x0, y0), (x1, y1), color, 2)

def draw_horizontal_lines(img, step=50):
    out = img.copy()
    for y in range(step, img.shape[0], step):
        cv.line(out, (0, y), (img.shape[1], y), (0, 0, 0), 1)
    return out

if __name__ == "__main__":
    CAMERA_PARAMETERS_LEFT = "camera_parameters_left_j.npz"
    CAMERA_PARAMETERS_RIGHT = "camera_parameters_right_j.npz"
    # CAMERA_PARAMETERS_LEFT = "camera_parameters_test_left.npz"
    # CAMERA_PARAMETERS_RIGHT = "camera_parameters_test_right.npz"
    square_size = 4 # in inch
    # square_size = 2 # in inch
    pattern_size = (10, 7) # dim of chess board, internal square intersections
    R, T, rvec, E, F = stereo_calibrate_from_files(left_folder=STEREO_L_IMG_FOLDER, right_folder=STEREO_R_IMG_FOLDER, 
                                             pattern_size=pattern_size, square_size=square_size, 
                                             CAMERA_PARAM_LEFT=CAMERA_PARAMETERS_LEFT, CAMERA_PARAM_RIGHT=CAMERA_PARAMETERS_RIGHT)
    
    ############################### TASK 3 ###############################
    # Load images
    left_image_path = os.path.join(STEREO_L_IMG_FOLDER, "0.png")
    right_image_path = os.path.join(STEREO_R_IMG_FOLDER, "0.png")
    imgL = cv.imread(left_image_path)
    imgR = cv.imread(right_image_path)

    ## camera parameters
    k_left, dist_left = load_camera_parameters(CAMERA_PARAMETERS_LEFT)
    k_right, dist_right = load_camera_parameters(CAMERA_PARAMETERS_RIGHT)

    # Undistort
    imgL_undist = cv.undistort(imgL, k_left, dist_left)
    imgR_undist = cv.undistort(imgR, k_right, dist_right)


    left_corners = find_corners(left_image_path, pattern_size)
    right_corners = find_corners(right_image_path, pattern_size)
    # corners shape: (N, 1, 2)
    cols, rows = pattern_size
    corners_L = left_corners.reshape(-1, 2)
    corners_R = right_corners.reshape(-1, 2)
    idx_tl = 0
    idx_tr = (rows - 3)* cols - 1
    idx_bl = (rows - 2) * cols
    idx_br = rows * cols - 1

    # get the points for left and right images, the four corners of chess board
    points_left = np.array([
        corners_L[idx_tl],
        corners_L[idx_tr],
        corners_L[idx_bl],
        corners_L[idx_br]
    ])
    points_right = np.array([
        corners_R[cols * 1 + 1],            # near top-left, but not extreme
        corners_R[cols * 2 + (cols - 2)],   # upper-right interior
        corners_R[cols * (rows - 3) + 1],   # lower-left interior
        corners_R[cols * (rows - 2) + (cols - 2)]  # lower-right interior
    ])
    
    draw_points(imgL_undist, points_left, (0,0,255)) # red
    draw_points(imgR_undist, points_right, (255,0,0)) # blue

    # Reshape for OpenCV (Nx1x2)
    ptsL = points_left.reshape(-1, 1, 2)
    ptsR = points_right.reshape(-1, 1, 2)

    # Epipolar lines
    lines_in_R = cv.computeCorrespondEpilines(ptsL, 1, F)
    lines_in_L = cv.computeCorrespondEpilines(ptsR, 2, F)

    lines_in_R = lines_in_R.reshape(-1,3)
    lines_in_L = lines_in_L.reshape(-1,3)

    draw_epilines(imgL_undist, lines_in_L, (255, 0, 0)) # blue
    draw_epilines(imgR_undist, lines_in_R, (0, 0, 255))  # red

    # plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Left image: points + epipolar lines from right")
    plt.imshow(cv.cvtColor(imgL_undist, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Right image: points + epipolar lines from left")
    plt.imshow(cv.cvtColor(imgR_undist, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.savefig("epipolar_lines.png")

    ####################### Task 4 ###############################
    left_image_path = os.path.join(STEREO_L_IMG_FOLDER, "0.png")
    right_image_path = os.path.join(STEREO_R_IMG_FOLDER, "0.png")
    imgL = cv.imread(left_image_path, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(right_image_path, cv.IMREAD_GRAYSCALE)
    h, w = imgL.shape[:2]
    image_size = (w, h)

    data = np.load(f"camera_params_stereo_j.npz")
    stereo_mtxL=data["stereo_mtxL"]
    stereo_distL=data["stereo_distL"]
    stereo_mtxR=data["stereo_mtxR"]
    stereo_distR=data["stereo_distR"]
    R=data["R"]
    T=data["T"]
    E=data["E"]
    F=data["F"]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        stereo_mtxL, stereo_distL,
        stereo_mtxR, stereo_distR,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(
        stereo_mtxL, stereo_distL, R1, P1,
        image_size, cv2.CV_32FC1
    )

    mapRx, mapRy = cv2.initUndistortRectifyMap(
        stereo_mtxR, stereo_distR, R2, P2,
        image_size, cv2.CV_32FC1
    )

    ############ Print some things ############
    print("Rectification rotation matrix R1 (3x3):\n", R1)
    print("Rectification rotation matrix R2 (3x3):\n", R2)
    # Left camera rectification rotation
    rvec1, _ = cv2.Rodrigues(R1)
    rvec1_deg = np.degrees(rvec1)

    # Right camera rectification rotation
    rvec2, _ = cv2.Rodrigues(R2)
    rvec2_deg = np.degrees(rvec2)

    print("R1 rotation vector (degrees):\n", rvec1_deg)
    print("R2 rotation vector (degrees):\n", rvec2_deg)

    print("P1:\n", P1)
    print("P2:\n", P2)

    print("Q\n:", Q)

    ################################################

    rectified_L = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectified_R = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    rectified_L = cv2.cvtColor(rectified_L, cv2.COLOR_GRAY2RGB)
    rectified_R = cv2.cvtColor(rectified_R, cv2.COLOR_GRAY2RGB)

    rectified_L_with_lines = draw_horizontal_lines(rectified_L.copy())
    rectified_R_with_lines = draw_horizontal_lines(rectified_R.copy())
    rectified_lines = np.hstack((rectified_L_with_lines, rectified_R_with_lines))
    key = show_image("Rectified", rectified_lines)
    cv2.imwrite("my_rectified_lines.png", rectified_lines)

    color_L = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
    color_R = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)

    orig = np.hstack((color_L, color_R))
    rectified = np.hstack((rectified_L, rectified_R))
    combined = np.vstack((orig, rectified))
    key = show_image("Orig, Rectified", combined)
    cv2.imwrite("my_orig.png", orig)
    cv2.imwrite("my_rectified.png", rectified)

    diffL = cv2.absdiff(color_L, rectified_L)
    diffR = cv2.absdiff(color_R, rectified_R)
    diff = np.hstack((diffL, diffR))
    key = show_image("Diffed", diff)
    cv2.imwrite("my_diff.png", diff)