# cspell:disable
import os
import sys
import cv2
import numpy as np

def load_params(side):
    data = np.load(f"camera_parameters_{side}_j.npz")
    mtx = data["camera_matrix"]
    dist = data["dist"]
    return mtx, dist

def load_image_names(side):
    image_dir = f"braden_image_calibration/{side}"
    # image_dir = f"practice_img/{side}"
    image_files = sorted(list(im for im in os.listdir(image_dir)))
    return image_files, image_dir

def find_corners(image_dir, file):
    frame = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(frame, (10, 7))
    if not ret:
        print("WARNING: corners not found")
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,     # max iterations
        0.001   # epsilon
    )

    # 3) Sub-pixel refinement
    corners = cv2.cornerSubPix(
        frame,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return corners

image_files_L, image_dir_L = load_image_names("SL")
image_files_R, image_dir_R = load_image_names("SR")

mtx_L, dist_L = load_params("left")
mtx_R, dist_R = load_params("right")

# prepare object points for corners
objp = np.zeros((10*7,3), np.float32)
square_size = 4 # 4x4 inches
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)*square_size
 
# Arrays to store obj points
objpoints = [] 
imgpoints_L = [] 
imgpoints_R = []

for file_L, file_R in zip(image_files_L, image_files_R):
    cornersL = find_corners(image_dir_L, file_L)
    cornersR = find_corners(image_dir_R, file_R)

    if cornersL is None or cornersR is None:
        print(f"Skipping pair: {file_L}, {file_R}")
        continue
    
    objpoints.append(objp)
    imgpoints_L.append(cornersL)
    imgpoints_R.append(cornersR)

frame = cv2.imread(os.path.join(image_dir_L, image_files_L[0]), cv2.IMREAD_GRAYSCALE)
image_size = frame.shape[::-1]

flags = cv2.CALIB_FIX_INTRINSIC
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5
)
ret, stereo_mtxL, stereo_distL, stereo_mtxR, stereo_distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_L,
    imgpoints_R,
    mtx_L,
    dist_L,
    mtx_R,
    dist_R,
    image_size,
    criteria=criteria,
    flags=flags
)

print("Rotation")
print(R)

rvec, _ = cv2.Rodrigues(R)  # rvec is 3x1 in radians
rvec_deg = np.degrees(rvec) # convert to degrees

print("Rotation vector (3x1) in degrees:\n", rvec_deg)
print("T")
print(T)
print("Essential Matrix")
print(E)
print("Fundamental Matrix")
print(F)

print("RMS error:", ret)

np.savez(f"camera_params_stereo.npz", 
         stereo_mtxL=stereo_mtxL,
         stereo_distL=stereo_distL,
         stereo_mtxR=stereo_mtxR,
         stereo_distR=stereo_distR,
         R=R,
         T=T,
         E=E,
         F=F)
