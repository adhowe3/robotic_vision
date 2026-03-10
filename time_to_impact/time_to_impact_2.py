import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# -----------------------------
# Camera Intrinsic Parameters
# -----------------------------
K = np.array([
    [825.0900600547, 0.0, 331.6538103208],
    [0.0, 824.2672147458, 252.9284287373],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    -0.2380769337,
    0.0931325835,
    0.0003242537,
    -0.0021901930,
    0.4641735616
])

# -----------------------------
# Known Parameters
# -----------------------------
gas_can_diameter_mm = 59
frame_translation_mm = 15.25   # distance camera moves each frame

# -----------------------------
# Load Images
# -----------------------------
image_paths = sorted(glob.glob("images/*.jpg"))

images = []
for p in image_paths:
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)

# -----------------------------
# Feature Detection
# -----------------------------
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Lucas Kanade tracker
lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

p0 = cv2.goodFeaturesToTrack(images[0], mask=None, **feature_params)

# store feature expansion measurements
expansion_rates = []
frame_numbers = []

prev_img = images[0]
prev_pts = p0

# -----------------------------
# Track features through frames
# -----------------------------
for i in range(1, len(images)):

    next_img = images[i]

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_img,
        next_img,
        prev_pts,
        None,
        **lk_params
    )

    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]

    # Compute radial expansion from center
    center = np.mean(good_old, axis=0)

    r_old = np.linalg.norm(good_old - center, axis=1)
    r_new = np.linalg.norm(good_new - center, axis=1)

    expansion = np.mean(r_new - r_old)

    expansion_rates.append(expansion)
    frame_numbers.append(i)

    prev_img = next_img.copy()
    prev_pts = good_new.reshape(-1,1,2)

# -----------------------------
# Task 2: Distance to impact
# Known velocity
# -----------------------------
distances = []

for i in range(len(images)):
    dist = (len(images) - i) * frame_translation_mm
    distances.append(dist)

plt.figure()
plt.plot(range(len(images)), distances)
plt.xlabel("Frame")
plt.ylabel("Distance (mm)")
plt.title("Distance vs Frame (Known Velocity)")
plt.savefig("distance_to_impact.png")