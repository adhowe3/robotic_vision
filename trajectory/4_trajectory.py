import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def read_and_crop_stereo(left_path, right_path, crop_box_l, crop_box_r):
    """
    Reads left and right images and returns cropped versions.

    crop_box = (x_start, x_end)
    """
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    if left_img is None:
        raise ValueError(f"Could not read left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Could not read right image: {right_path}")
    
    x_start, x_end = crop_box_l
    left_crop = left_img[:, x_start:x_end]
    x_start, x_end = crop_box_r # different crop for right images
    right_crop = right_img[:, x_start:x_end]

    return left_crop, right_crop

def differencing(image, prev_ref_gray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_ref_gray)
    return diff

def contours(diff, original):
    # 1. Blur first
    diff_blur = cv2.GaussianBlur(diff, (7,7), 0)
    # 2. Threshold
    _, mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)
    # 3. Morphological close (fills holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 4. Find contours
    contours_list, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours_list:
        print("no contours found")
        return original, (0,0)

    # 5. Select largest contour
    largest = max(contours_list, key=cv2.contourArea)
    
    # 6. Compute enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)
    # 7. Draw result
    output = original.copy()
    # output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cv2.circle(output, center, radius, (0,255,255), 2)
    cv2.circle(output, center, 3, (0,0,255), -1)
    return output, center 

def detect_baseball(image, prev_ref_gray):
    diff = differencing(image, prev_ref_gray)
    cont, center = contours(diff, image)
    return cont, center

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

def compute_global_scale(z_points, x_points, y_points):
    z = np.array(z_points)
    x = np.array(x_points)
    y = np.array(y_points)

    z_range = max(z) - min(z)
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)

    return z_range, max(x_range, y_range)+80

def plot_z_x(z_points, x_points, z_range, x_range):
    z = np.array(z_points)
    x = np.array(x_points)

    # ---- LINEAR FIT (Degree 1) ----
    # This finds the 'm' and 'b' in x = mZ + b
    coeffs = np.polyfit(z, x, 1) 
    p = np.poly1d(coeffs)

    # 1. Generate fit line from the back of the range to 0
    z_fit = np.linspace(z_range, 0, 400)
    x_fit = p(z_fit)

    plt.figure()
    
    # Plot predicted path (straight line)
    plt.plot(z_fit, x_fit, linewidth=2, color='red', label='Predicted Path')
    
    # Plot ball locations as hollow circles
    plt.scatter(z, x, s=50, facecolors='none', edgecolors='blue', 
                linewidths=1.5, label='Ball Trajectory')

    # 2. Mark the intercept at Z=0
    x_at_zero = p(0)

    # 3. Viewport Control
    plt.xlim(z_range, 0) # Z=0 on the right
    
    x_mid = (max(x) + min(x)) / 2
    plt.ylim(x_mid + x_range/2, x_mid - x_range/2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Z (Depth)")
    plt.ylabel("X (Horizontal)")
    plt.title("Top Down View: Path to Z=0")
    
    plt.savefig("z_x.png")
    plt.close()

    return x_at_zero


def plot_z_y(z_points, y_points, z_range, y_range):
    z = np.array(z_points)
    y = np.array(y_points)

    coeffs = np.polyfit(z, y, 2)
    p = np.poly1d(coeffs)

    # 1. Generate fit line from the very back of the range all the way to 0
    # This ensures the line exists even where there are no dots
    z_fit = np.linspace(z_range, 0, 400)
    y_fit = p(z_fit)

    plt.figure()
    plt.plot(z_fit, y_fit, linewidth=2, color='red', label='Predicted Path')
    plt.scatter(z, y, s=30, facecolors='none', edgecolors='blue', label='Ball Trajectory')

    # 2. Explicitly plot the intercept at Z=0
    y_at_zero = p(0)
    
    # 3. Force the Viewport to end at 0
    # Left side is the range, Right side is 0
    plt.xlim(z_range, 0) 
    
    y_mid = (max(y) + min(y)) / 2
    plt.ylim(y_mid + y_range/2, y_mid - y_range/2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Z (Depth)")
    plt.ylabel("Y (Vertical)")
    plt.title("Side View: Path to Z=0")
    
    plt.savefig("z_y.png")
    plt.close()

    return y_at_zero


################# main ######################
if __name__ == "__main__":
    base_path = "ball_images"
    out_base_path = "out_ball_images"
    crop_box_l = (300, 550)
    crop_box_r = (100, 350)
    left_folder = os.path.join(base_path, "L")
    right_folder = os.path.join(base_path, "R")

    # sort numerically (0,1,2 etc)
    left_files = sorted(
        [f for f in os.listdir(left_folder) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    right_files = sorted(
        [f for f in os.listdir(right_folder) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    first_left_img = 0
    first_right_img = 0

    # save the output images to see
    os.makedirs(os.path.join(out_base_path, "L"), exist_ok=True)
    os.makedirs(os.path.join(out_base_path, "R"), exist_ok=True)

    plot_points_x = []
    plot_points_y = []
    plot_points_z = []

    ## load the npz file ##
    data = np.load("camera_params_stereo.npz")
    R = data["R"]
    T = data["T"]
    baseline_camera_dist = abs(T[0])
    print(T)

    test_image = cv2.imread("ball_images/L/0.png")
    h, w, c = test_image.shape
    image_size = (w, h)
    print("original image_size: ", image_size)

    for l_file, r_file in zip(left_files, right_files):

        left_path = os.path.join(left_folder, l_file)
        right_path = os.path.join(right_folder, r_file)

        out_left_folder = os.path.join(out_base_path, "L", l_file)
        out_right_folder = os.path.join(out_base_path, "R", r_file)

        # crop the left and right images
        left_img, right_img = read_and_crop_stereo(left_path, right_path, crop_box_l, crop_box_r)
        
        # get first image in set saved
        if(l_file == "0.png" and r_file == "0.png"):
            first_left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            first_right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        # detect baseball
        else:
            left_img_detected, left_points = detect_baseball(left_img, first_left_img)
            right_img_detected, right_points = detect_baseball(right_img, first_right_img)
            if left_points == (0,0) or right_points == (0,0):
                continue
            # cv2.imwrite(out_left_folder, left_img_detected)
            # cv2.imwrite(out_right_folder, right_img_detected)

            # uncrop the left and right poitns
            left_points_uncropped = crop_box_l[0] + left_points[0], left_points[1]
            right_points_uncropped = crop_box_r[0] + right_points[0], right_points[1]
            # print("left: ", left_points, "uncropped: ", left_points_uncropped)
            # print("right: ", right_points, "uncropped: ", right_points_uncropped)

            # undistort and rectify the points, triangulate 3d points
            rectL = undistort_and_rectify_corners(data, image_size, left_points_uncropped, side="L")
            rectR = undistort_and_rectify_corners(data, image_size, right_points_uncropped, side="R")
            left_3d = triangulate(data, image_size, rectL, rectR)
            # print("left: ", left_3d)

            half_baseline = baseline_camera_dist / 2.0
            # Shift X so midpoint becomes origin
            catcher_3d = left_3d.copy()
            catcher_3d[:, 0] = catcher_3d[:, 0] + half_baseline
            plot_points_x.append(catcher_3d[0,0])
            plot_points_y.append(catcher_3d[0,1])
            plot_points_z.append(catcher_3d[0,2])

    # plot the path
    z_range, y_range = compute_global_scale(
        plot_points_z,
        plot_points_x,
        plot_points_y
    )

    # pass in y_range to both to keep plots scaled the same
    x_zero = plot_z_x(plot_points_z, plot_points_x, z_range, y_range)
    y_zero = plot_z_y(plot_points_z, plot_points_y, z_range, y_range)

    print("Predicted intercept at Z=0:")
    print("X =", x_zero)
    print("Y =", y_zero)
        


