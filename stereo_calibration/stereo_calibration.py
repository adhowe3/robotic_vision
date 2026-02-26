import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

INPUT_IMG_FOLDER = "./input_images"
OUTPUT_IMG_FOLDER = "./output_images"
STEREO_R_IMG_FOLDER = "./braden_image_calibration/SR"
STEREO_L_IMG_FOLDER = "./braden_image_calibration/SL"

PRACTICE_SL = "practice_img/SL"
PRACTICE_SR = "practice_img/SR"

def rotationMatrixToEulerXYZ(R):
    """
    Convert rotation matrix to XYZ Euler angles
    Returns angles in radians
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


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

    flags = 0
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

    # calculate the stereoCalibrate values
    rms, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        object_points,
        image_points_L,
        image_points_R,
        K1, dist1,
        K2, dist2,
        image_size,
        criteria=criteria,
        flags=flags
    )
    rvec, _ = cv.Rodrigues(R)
    euler_rad = rotationMatrixToEulerXYZ(R)
    euler_deg = np.degrees(euler_rad)

    print("Stereo RMS error:", rms)
    print("\nRotation matrix (3x3):")
    print(R)
    print("\nTranslation vector T (3x1):")
    print(T)
    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)
    print("\nRotation vector (3x1) [radians]:")
    print(rvec)
    print("\nEuler angles (XYZ order):")
    print(f"Rotation about X: {euler_deg[0]:.6f}°")
    print(f"Rotation about Y: {euler_deg[1]:.6f}°")
    print(f"Rotation about Z: {euler_deg[2]:.6f}°")

    return R, rvec, euler_deg, T, E, F


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
        cv.line(out, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    return out

if __name__ == "__main__":
    CAMERA_PARAMETERS_LEFT = "camera_parameters_left.npz"
    CAMERA_PARAMETERS_RIGHT = "camera_parameters_right.npz"
    # CAMERA_PARAMETERS_LEFT = "camera_parameters_test_left.npz"
    # CAMERA_PARAMETERS_RIGHT = "camera_parameters_test_right.npz"
    square_size = 3.985 # in inch
    # square_size = 2 # in inch
    pattern_size = (10, 7) # dim of chess board, internal square intersections
    R, T, rvec, angles_deg, E, F = stereo_calibrate_from_files(left_folder=STEREO_L_IMG_FOLDER, right_folder=STEREO_R_IMG_FOLDER, 
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
    idx_tr = cols - 1
    idx_bl = (rows - 1) * cols
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

    draw_epilines(imgR_undist, lines_in_R, (0, 255, 0))  # green
    draw_epilines(imgL_undist, lines_in_L, (0, 255, 0))

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

    ##################### Task 4 ###############################
    h, w = imgL.shape[:2]
    image_size = (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        k_left, dist_left,
        k_right, dist_right,
        image_size,
        R, T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=1
    )

    mapL1, mapL2 = cv.initUndistortRectifyMap(
        k_left, dist_left, R1, P1, image_size, cv.CV_16SC2
    )

    mapR1, mapR2 = cv.initUndistortRectifyMap(
        k_right, dist_right, R2, P2, image_size, cv.CV_16SC2
    )

    rectL = cv.remap(imgL, mapL1, mapL2, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, mapR1, mapR2, cv.INTER_LINEAR)

    rectL_lines = draw_horizontal_lines(rectL)
    rectR_lines = draw_horizontal_lines(rectR)


    diffL = cv.absdiff(rectL, imgL)
    diffR = cv.absdiff(rectR, imgR)

    # save figs
    cv.imwrite("original_left.png", imgL)
    cv.imwrite("original_right.png", imgR)

    cv.imwrite("rectified_left_lines.png", rectL_lines)
    cv.imwrite("rectified_right_lines.png", rectR_lines)

    cv.imwrite("absdiff_left.png", diffL)
    cv.imwrite("absdiff_right.png", diffR)

    # rotation stuff
    Rrect = R1  # Left rectification rotation matrix
    print("\nRectification Rotation Matrix Rrect (3x3):")
    print(Rrect)
    rvec_rect, _ = cv.Rodrigues(Rrect)
    euler_rad = rotationMatrixToEulerXYZ(Rrect)
    euler_deg = np.degrees(euler_rad)

    print("\nRectification rotation vector rvec (3x1) [radians]:")
    print(rvec_rect)

    print("\nRectification rotation about X, Y, Z [degrees]:")
    print(f"Rx: {euler_deg[0]:.6f}°")
    print(f"Ry: {euler_deg[1]:.6f}°")
    print(f"Rz: {euler_deg[2]:.6f}°")
