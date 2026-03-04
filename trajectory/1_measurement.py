import cv2
import numpy as np
import os

pattern_size = (10, 7)  # inner corners (columns, rows) UNCOMMENT FOR TASK_2()
square_size = 4  # this is in inches (the EB large board)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

# keep the start-end index images, remove the rest from the dataset 
def keep_image_range(base_path, start, end):
    for side in ["L", "R"]:
        folder = os.path.join(base_path, side)

        for file in os.listdir(folder):
            if file.endswith(".png"):
                idx = int(file.split(".")[0])

                if idx < start or idx > end:
                    os.remove(os.path.join(folder, file))


def get_chess_board_corners(file_name, input_path=".", output_path="output"):
    infile = os.path.join(input_path, file_name)
    outfile = os.path.join(output_path, file_name)
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(infile)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, pattern_size)
    
    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        win_size = (5, 5)
        zero_zone = (-1, -1)
        corners_refined = cv2.cornerSubPix(gray_image, corners, win_size, zero_zone, criteria)
        
        # Draw corners on a color version of the grayscale image
        output_img = image.copy()
        for corner in corners_refined:
            x, y = corner.ravel()
            cv2.circle(output_img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green
        
        # Show the image
        cv2.imwrite(outfile, output_img)
        return corners_refined
    else:
        print("Chessboard corners not found! - ", infile)


def undistort_and_rectify_corners(data, image_size, corners, side="L"):
    """
    data: npz file
    image_size: (width, height)
    corners: Nx2 or Nx1x2 array of distorted corner points
    side: "L" or "R"
    """

    mtxL = data["stereo_mtxL"]
    distL = data["stereo_distL"]
    mtxR = data["stereo_mtxR"]
    distR = data["stereo_distR"]
    R = data["R"]
    T = data["T"]

    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL,
        mtxR, distR,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # Ensure correct shape for undistortPoints
    corners = np.asarray(corners, dtype=np.float32)
    if corners.ndim == 2:
        corners = corners.reshape(-1, 1, 2)

    if side == "L":
        rectified = cv2.undistortPoints(
            corners, mtxL, distL, R=R1, P=P1
        )
    else:
        rectified = cv2.undistortPoints(
            corners, mtxR, distR, R=R2, P=P2
        )

    return rectified.reshape(-1, 2)

def get_outer_corners(corners):
    """
    corners: output of get_chess_board_corners()  (Nx1x2 or Nx2)
    """
    cols, rows = pattern_size
    corners = corners.reshape(-1, 2)

    top_left = corners[0]
    top_right = corners[cols - 1]
    bottom_left = corners[(rows - 1) * cols]
    bottom_right = corners[-1]

    return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)


def draw_before_after_corners(image, original_pts, rectified_pts, output_path):
    """
    image: original distorted image (cv2.imread)
    original_pts: Nx2 original distorted points
    rectified_pts: Nx2 rectified points
    output_path: where to save result
    """
    img = image.copy()

    original_pts = np.asarray(original_pts).reshape(-1, 2)
    rectified_pts = np.asarray(rectified_pts).reshape(-1, 2)

    # Draw original points in GREEN
    for (x, y) in original_pts:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Draw rectified points in RED
    for (x, y) in rectified_pts:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)


def triangulate(data, image_size, rectified_pts_L, rectified_pts_R):
    """
    data: loaded npz file containing stereo parameters
    image_size: (width, height)
    rectified_pts_L: Nx2 rectified points from left image
    rectified_pts_R: Nx2 rectified points from right image

    Returns:
        points3D: Nx3 array of triangulated 3D points
    """

    mtxL = data["stereo_mtxL"]
    distL = data["stereo_distL"]
    mtxR = data["stereo_mtxR"]
    distR = data["stereo_distR"]
    R = data["R"]
    T = data["T"]

    # Compute rectification projection matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL,
        mtxR, distR,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # Ensure correct shape: 2xN
    ptsL = np.asarray(rectified_pts_L, dtype=np.float32).T
    ptsR = np.asarray(rectified_pts_R, dtype=np.float32).T

    # Triangulate (returns 4xN homogeneous coordinates)
    points4D = cv2.triangulatePoints(P1, P2, ptsL, ptsR)

    # Convert from homogeneous to Euclidean coordinates
    points3D = points4D[:3] / points4D[3]

    # Return as Nx3
    t_points3D = points3D.T
    # compute_reprojection_error(P1, P2, t_points3D, ptsL.T, ptsR.T)
    return t_points3D


def compute_reprojection_error(P1, P2, points3D, ptsL, ptsR):

    N = points3D.shape[0]
    print(N)
    points4D = np.hstack((points3D, np.ones((N,1))))

    projL = (P1 @ points4D.T).T
    projR = (P2 @ points4D.T).T

    projL = projL[:, :2] / projL[:, 2:3]
    projR = projR[:, :2] / projR[:, 2:3]

    errL = np.linalg.norm(projL - ptsL, axis=1)
    errR = np.linalg.norm(projR - ptsR, axis=1)

    print("Mean reprojection error Left:", np.mean(errL))
    print("Mean reprojection error Right:", np.mean(errR))

    return np.mean(errL), np.mean(errR)


def convert_left_to_right(points3D_left, R, T):
    """
    Convert 3D points from left camera frame
    to right camera frame
    """
    points3D_left = np.asarray(points3D_left)
    points3D_right = []
    for P in points3D_left:
        Pr = R.T @ (P - T.flatten())
        points3D_right.append(Pr)

    return np.array(points3D_right)


def verify_rigid_transform(points_left, points_right, R, T):
    predicted_left = []
    for Pr in points_right:
        Pl_pred = R @ Pr + T.flatten()
        predicted_left.append(Pl_pred)

    predicted_left = np.array(predicted_left)
    error = np.linalg.norm(predicted_left - points_left, axis=1)
    print("Rigid transform verification error:")
    print(error)
    print("Mean error:", np.mean(error))

################# main ######################
if __name__ == "__main__":
    # keep_image_range("ball_images", 21, 65) # remove images 0-20 and 65-99
    left_corners = get_chess_board_corners("L_0.png")
    right_corners = get_chess_board_corners("R_0.png")
    left_outer_corners = get_outer_corners(left_corners)
    right_outer_corners = get_outer_corners(right_corners)

    imgL = cv2.imread("L_0.png")
    imgR = cv2.imread("R_0.png")

    first_image = cv2.imread("L_0.png")
    h, w, c = first_image.shape
    image_size = (w, h)
    print(image_size)

    ## load the npz file ##
    data = np.load("camera_params_stereo.npz")
    mtxL = data["stereo_mtxL"]
    distL = data["stereo_distL"]
    mtxR = data["stereo_mtxR"]
    distR = data["stereo_distR"]
    R = data["R"]
    T = data["T"]

    rectL = undistort_and_rectify_corners(data, image_size, left_outer_corners, side="L")
    rectR = undistort_and_rectify_corners(data, image_size, right_outer_corners, side="R")
    
    draw_before_after_corners(
        imgL,
        left_outer_corners,
        rectL,
        "left_before_after.png"
    )

    draw_before_after_corners(
        imgR,
        right_outer_corners,
        rectR,
        "right_before_after.png"
    )

    ## triangulate points ##
    points3D_left = triangulate(data, image_size, rectL, rectR)

    points3D_right = convert_left_to_right(points3D_left, R, T)

    print("3D points w.r.t LEFT camera:")
    print(points3D_left)

    print("3D points w.r.t RIGHT camera:")
    print(points3D_right)
    verify_rigid_transform(points3D_left, points3D_right, R, T)