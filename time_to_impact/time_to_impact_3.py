import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

fx = 825.0900600547
object_width_mm = 59

left_feature = 133
right_feature = 150

os.makedirs("output3", exist_ok=True)

# -----------------------------
# Load images
# -----------------------------
image_paths = glob.glob("images/T*.jpg")

def frame_number(path):
    return int(re.search(r'\d+', path).group())

image_paths = sorted(image_paths, key=frame_number)

images = []
for p in image_paths:
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append((img, gray))

# -----------------------------
# Detect features in first frame
# -----------------------------
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)

first_color, first_gray = images[0]

p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

# select chosen features
selected_pts = np.array([
    p0[left_feature],
    p0[right_feature]
])

# -----------------------------
# Lucas Kanade parameters
# -----------------------------
lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

prev_gray = first_gray
prev_pts = selected_pts

frames = []
pixel_widths = []

# -----------------------------
# Track features
# -----------------------------
for i in range(1, len(images)):

    color, gray = images[i]

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        **lk_params
    )

    p_left = next_pts[0].ravel()
    p_right = next_pts[1].ravel()

    width_pixels = np.linalg.norm(p_left - p_right)

    frames.append(i)
    pixel_widths.append(width_pixels)

    # visualization
    vis = color.copy()

    cv2.circle(vis, tuple(p_left.astype(int)), 6, (0,255,0), -1)
    cv2.circle(vis, tuple(p_right.astype(int)), 6, (0,255,0), -1)

    cv2.line(vis,
             tuple(p_left.astype(int)),
             tuple(p_right.astype(int)),
             (255,0,0),2)

    cv2.putText(vis,
                f"Width: {width_pixels:.2f}px",
                (40,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2)

    cv2.imwrite(f"output3/track_{i:03d}.png", vis)

    prev_gray = gray.copy()
    prev_pts = next_pts

# -----------------------------
# Convert pixel width -> distance
# -----------------------------
distances = []

for w in pixel_widths:

    Z = (fx * object_width_mm) / w

    distances.append(Z)

# -----------------------------
# Plot distance vs frame
# -----------------------------
plt.figure()

plt.plot(frames, distances, marker='o')

plt.xlabel("Frame Number")
plt.ylabel("Distance to Object (mm)")
plt.title("Object Distance vs Frame (Known Object Size & Camera Parameters)")

plt.grid()

plt.savefig("distance_vs_frame_3.png")

print("Saved tracking images and distance plot to output3/")
print("distance from last image: ", distances[-1], "mm")