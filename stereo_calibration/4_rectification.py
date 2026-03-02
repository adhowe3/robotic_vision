# cspell:disable
import os
import cv2
import numpy as np

def load_image_names(side):
    image_dir = f"braden_image_calibration/{side}"
    image_files = sorted(list(im for im in os.listdir(image_dir)))
    return image_files, image_dir

def draw_h_lines(img):
    h, w, c = img.shape
    color=(0,255,0)    
    for i in range(50, h, 50):
        x0, y0 = 0, i
        x1, y1 = w, i
        cv2.line(img, (x0,y0), (x1,y1), color, 1)
    return img

def show_image(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key

img_filesL, img_dirL = load_image_names("SL")
img_filesR, img_dirR = load_image_names("SR")

data = np.load(f"camera_params_stereo.npz")
stereo_matrixL=data["stereo_mtxL"]
stereo_distL=data["stereo_distL"]
stereo_matrixR=data["stereo_mtxR"]
stereo_distR=data["stereo_distR"]
R=data["R"]
T=data["T"]
E=data["E"]
F=data["F"]

for file_L, file_R in zip(img_filesL, img_filesR):
    imgL = cv2.imread(os.path.join(img_dirL, file_L), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(img_dirR, file_R), cv2.IMREAD_GRAYSCALE)
    image_size = imgL.shape[::-1]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        stereo_matrixL, stereo_distL,
        stereo_matrixR, stereo_distR,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    mapLeft_x, mapLeft_y = cv2.initUndistortRectifyMap(stereo_matrixL, stereo_distL, R1, P1,image_size, cv2.CV_32FC1)
    mapRight_x, mapRight_y = cv2.initUndistortRectifyMap(stereo_matrixR, stereo_distR, R2, P2, image_size, cv2.CV_32FC1)

    ############ Print matricies ############
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

    ######################### Do some image saving here #######################

    rectified_L = cv2.remap(imgL, mapLeft_x, mapLeft_y, cv2.INTER_LINEAR)
    rectified_R = cv2.remap(imgR, mapRight_x, mapRight_y, cv2.INTER_LINEAR)

    rectified_L = cv2.cvtColor(rectified_L, cv2.COLOR_GRAY2RGB)
    rectified_R = cv2.cvtColor(rectified_R, cv2.COLOR_GRAY2RGB)

    rectified_L_with_lines = draw_h_lines(rectified_L.copy())
    rectified_R_with_lines = draw_h_lines(rectified_R.copy())
    rectified_lines = np.hstack((rectified_L_with_lines, rectified_R_with_lines))
    key = show_image("Rectified", rectified_lines)
    if key == 27:  # press ESC to quit
        break
    elif key == ord('s'):   # press S to save
        cv2.imwrite("rectified_lines.png", rectified_lines)

    color_L = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
    color_R = cv2.cvtColor(imgR, cv2.COLOR_GRAY2RGB)

    orig = np.hstack((color_L, color_R))
    rectified = np.hstack((rectified_L, rectified_R))
    combined = np.vstack((orig, rectified))
    key = show_image("Orig, Rectified", combined)
    if key == 27:  # press ESC to quit
        break
    elif key == ord('s'):
        cv2.imwrite("orig.png", orig)
        cv2.imwrite("rectified.png", rectified)

    diffL = cv2.absdiff(color_L, rectified_L)
    diffR = cv2.absdiff(color_R, rectified_R)
    diff = np.hstack((diffL, diffR))
    key = show_image("Diffed", diff)
    if key == 27:  # press ESC to quit
        break
    elif key == ord('s'):
        cv2.imwrite("diff.png", diff)
