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
    print("HERE1")
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

    print("HERE2")
    # Image size
    sample_img = cv.imread(os.path.join(left_folder, left_images[0]))
    h, w = sample_img.shape[:2]
    image_size = (w, h)

    flags = cv.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
    print("HERE3")

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
    print("HERE4")

        # --- Rotation conversions ---
    # rvec, _ = cv.Rodrigues(R)
    # angle = np.linalg.norm(rvec)
    # axis = rvec.flatten() / angle
    # print("Rotation angle (deg):", np.degrees(angle))
    # print("Rotation axis:", axis)
    angle = 1 # TO DO FIX
    rvec = 1

    print("Stereo RMS error:", rms)

    print("\nRotation matrix (3x3):")
    print(R)

    print("\nRotation vector (3x1) [radians]:")
    print(rvec)

    print("\nTranslation vector T (3x1):")
    print(T)

    print("Essential matrix E:\n", E)
    print("Fundamental matrix F:\n", F)

    return R, rvec, angle, T, E, F


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
    h = out.shape[0]
    for y in range(step, h, step):
        cv.line(out, (0, y), (out.shape[1], y), (0, 255, 0), 1)
    return out


if __name__ == "__main__":
    CAMERA_PARAMETERS_LEFT = "camera_parameters_left.npz"
    CAMERA_PARAMETERS_RIGHT = "camera_parameters_right.npz"
    # CAMERA_PARAMETERS_LEFT = "camera_parameters_test_left.npz"
    # CAMERA_PARAMETERS_RIGHT = "camera_parameters_test_right.npz"
    square_size = 101.28 # in mm
    # square_size = 50.8 # in mm
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
    imgL = cv.imread(left_image_path)
    imgR = cv.imread(right_image_path)

    h, w = imgL.shape[:2]
    image_size = (w, h)
    ## camera parameters
    k_left, dist_left = load_camera_parameters(CAMERA_PARAMETERS_LEFT)
    k_right, dist_right = load_camera_parameters(CAMERA_PARAMETERS_RIGHT)

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        k_left, dist_left,
        k_right, dist_right,
        image_size,
        R, T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapLx, mapLy = cv.initUndistortRectifyMap(
        k_left, dist_left, R1, P1, image_size, cv.CV_32FC1
    )

    mapRx, mapRy = cv.initUndistortRectifyMap(
        k_right, dist_right, R2, P2, image_size, cv.CV_32FC1
    )

    imgL_rect = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)
    imgR_rect = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)

    imgL_rect_lines = draw_horizontal_lines(imgL_rect)
    imgR_rect_lines = draw_horizontal_lines(imgR_rect)
    cv.imwrite("imgL_rect_lines.png", imgL_rect_lines)
    cv.imwrite("imgR_rect_lines.png", imgR_rect_lines)
    cv.imwrite("imgL_rect.png", imgL_rect)
    cv.imwrite("imgR_rect.png", imgR_rect)

    diffL = cv.absdiff(imgL_rect, imgL)
    diffR = cv.absdiff(imgR_rect, imgR)
    cv.imwrite("diffL.png", diffL)
    cv.imwrite("diffR.png", diffR)

    print("Rrect (3x3):\n", R1)
    rvec_rect, _ = cv.Rodrigues(R1)

    rx, ry, rz = np.degrees(rvec_rect.flatten())

    print("Rrect (3x1) [radians]:\n", rvec_rect)
    print(f"Rotation about X, Y, Z [degrees]:\nRx={rx:.6f}, Ry={ry:.6f}, Rz={rz:.6f}")