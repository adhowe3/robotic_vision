import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os

fx = 825.0900600547
object_width_mm = 59

os.makedirs("output3", exist_ok=True)

image_paths = glob.glob("images/T*.jpg")

def frame_number(path):
    return int(re.search(r'\d+', path).group())

image_paths = sorted(image_paths, key=frame_number)

distances = []
frames = []

for i, path in enumerate(image_paths):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # moderate blur
    # blur = cv2.GaussianBlur(gray, (11,11), 0)
    # cv2.imwrite(f"output3/T{i+1}_blur.png", blur)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"output3/T{i+1}_bw.png", bw)

    # sobel vertical edges (strong for can sides)
    sobelx = cv2.Sobel(bw, cv2.CV_64F, 1, 0, ksize=5)

    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(sobelx)

    cv2.imwrite(f"output3/T{i+1}_sobel.png", sobelx)

    # threshold to get edges
    _, edges = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f"output3/T{i+1}_edges.png", edges)

    # -----------------------------
    # Morphological closing
    # -----------------------------
    kernel = np.ones((15,15), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        continue

    # filter by area (ignore tiny text contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largest = contours[0]

    x,y,w,h = cv2.boundingRect(largest)

    Z = (fx * object_width_mm) / w

    distances.append(Z)
    frames.append(i)

    # -----------------------------
    # Visualization
    # -----------------------------
    vis = img.copy()

    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.putText(
        vis,
        f"W={w}px  Z={Z:.1f}mm",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imwrite(f"output2/T{i+1}_detect.png", vis)

    print(f"Frame {i}: width={w}px distance={Z:.2f} mm")

# -----------------------------
# Plot distance vs frame
# -----------------------------
plt.figure()

plt.plot(frames, distances, marker='o')

plt.xlabel("Frame Number")
plt.ylabel("Distance to Object (mm)")
plt.title("Object Distance vs Frame (Known Object Size)")

plt.grid()

plt.savefig("time_to_impact_3.png")