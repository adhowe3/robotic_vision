# cspell:disable
import os
import sys
import cv2
import numpy as np

def load_params(side):
    data = np.load(f"camera_parameters_{side}.npz")
    mtx = data["camera_matrix"]
    dist = data["dist"]
    return mtx, dist


def load_image_names(side):
    # image_dir = f"practice_img/{side}"
    image_dir = f"braden_image_calibration/{side}"
    image_files = sorted(list(im for im in os.listdir(image_dir)))
    return image_files, image_dir


def draw_epilines(img, lines, color=(0,255,0)):
    r, c = img.shape[:2]
    for rline in lines:
        a, b, c_line = rline
        # y = (-c - ax) / b
        x0, y0 = 0, int(-c_line / b)
        x1, y1 = c-1, int(-(c_line + a*(c-1))/b)
        cv2.line(img, (x0,y0), (x1,y1), color, 1)
    return img


def get_random_points(img):
    H, W = img.shape[:2]
    num_points = 4
    # points on left side of image
    ptsR = np.zeros((num_points, 2), dtype=np.float32)
    ptsR[:, 0] = np.random.randint(W/5, W/2, size=num_points)  # x coordinates
    ptsR[:, 1] = np.random.randint(H/5, H-H/5, size=num_points)  # y coordinates

    # points on right side of image
    ptsL = np.zeros((num_points, 2), dtype=np.float32)
    ptsL[:, 0] = np.random.randint(W/2, W-W/5, size=num_points)  # x coordinates
    ptsL[:, 1] = np.random.randint(H/5, H-H/5, size=num_points)  # y coordinates

    return ptsL, ptsR


image_files_L, image_dir_L = load_image_names("SL")
image_files_R, image_dir_R = load_image_names("SR")

mtx_L, dist_L = load_params("left")
mtx_R, dist_R = load_params("right")

# load parameters
data = np.load(f"camera_params_stereo.npz")
stereo_mtxL=data["stereo_mtxL"]
stereo_distL=data["stereo_distL"]
stereo_mtxR=data["stereo_mtxR"]
stereo_distR=data["stereo_distR"]
R=data["R"]
T=data["T"]
E=data["E"]
F=data["F"]

# loop through data set and save images
for file_L, file_R in zip(image_files_L, image_files_R):
    if "22" in file_L:
        continue

    imgL = frame = cv2.imread(os.path.join(image_dir_L, file_L), cv2.IMREAD_GRAYSCALE)
    imgR = frame = cv2.imread(os.path.join(image_dir_R, file_R), cv2.IMREAD_GRAYSCALE)
    imgL_undist = cv2.undistort(imgL, mtx_L, dist_L)
    imgR_undist = cv2.undistort(imgR, mtx_R, dist_R)

    ptsL, ptsR = get_random_points(imgL_undist)

    # Draw circles for visualization
    color_L = cv2.cvtColor(imgL_undist, cv2.COLOR_GRAY2RGB)
    color_R = cv2.cvtColor(imgR_undist, cv2.COLOR_GRAY2RGB)
    for pt in ptsL:
        cv2.circle(color_L, tuple(pt.astype(int)), 5, (0,0,255), -1)  # Red
    for pt in ptsR:
        cv2.circle(color_R, tuple(pt.astype(int)), 5, (255,0,0), -1)  # Blue
    
    lines_in_L = cv2.computeCorrespondEpilines(ptsR.reshape(-1,1,2), 2, F)
    lines_in_R = cv2.computeCorrespondEpilines(ptsL.reshape(-1,1,2), 1, F)

    lines_in_L = lines_in_L.reshape(-1,3)
    lines_in_R = lines_in_R.reshape(-1,3)
    imgL_with_lines = draw_epilines(color_L.copy(), lines_in_L, color=(255,0,0))
    imgR_with_lines = draw_epilines(color_R.copy(), lines_in_R, color=(0,0,255))

    combined = np.hstack((imgL_with_lines, imgR_with_lines))
    cv2.imshow("Epipolar Lines (L | R)", combined)
    key = cv2.waitKey(0)
    if key == 27:  # press ESC to quit
        break
    elif key == ord('s'):
        cv2.imwrite("epipole.png", combined)
    cv2.destroyAllWindows()