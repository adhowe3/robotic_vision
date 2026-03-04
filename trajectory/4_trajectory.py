import cv2
import numpy as np
import os

def read_and_crop_stereo(left_path, right_path, crop_box_l, crop_box_r):
    """
    Reads left and right images and returns cropped versions.

    crop_box = (x_start, x_end, y_start, y_end)
    """
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    if left_img is None:
        raise ValueError(f"Could not read left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Could not read right image: {right_path}")
    
    x_start, x_end, y_start, y_end = crop_box_l
    left_crop = left_img[y_start:y_end, x_start:x_end]
    x_start, x_end, y_start, y_end = crop_box_r # different crop for right images
    right_crop = right_img[y_start:y_end, x_start:x_end]

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

################# main ######################
if __name__ == "__main__":
    base_path = "ball_images"
    out_base_path = "out_ball_images"
    crop_box_l = (300, 550, 0, 700)
    crop_box_r = (100, 350, 0, 700)
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

    for l_file, r_file in zip(left_files, right_files):

        left_path = os.path.join(left_folder, l_file)
        right_path = os.path.join(right_folder, r_file)

        out_left_folder = os.path.join(out_base_path, "L", l_file)
        out_right_folder = os.path.join(out_base_path, "R", r_file)

        # crop the left and right images
        left_img, right_img = read_and_crop_stereo(left_path, right_path, crop_box_l, crop_box_r)

        # detect baseball
        if(l_file == "0.png" and r_file == "0.png"):
            first_left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            first_right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_img_detected, left_points = detect_baseball(left_img, first_left_img)
            right_img_detected, right_points = detect_baseball(right_img, first_right_img)
            cv2.imwrite(out_left_folder, left_img_detected)
            cv2.imwrite(out_right_folder, right_img_detected)

