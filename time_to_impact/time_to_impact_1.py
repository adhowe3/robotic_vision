import cv2
import numpy as np
import glob
import os
import re

def compute_foe(p_old, p_new):
    A = []
    b = []

    for (x1,y1),(x2,y2) in zip(p_old, p_new):

        u = x2 - x1
        v = y2 - y1

        A.append([v, -u])
        b.append(v*x1 - u*y1)

    A = np.array(A)
    b = np.array(b)

    # least squares solution
    foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return foe

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("output", exist_ok=True)

# -----------------------------
# Load images in numeric order
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
# Feature detection parameters
# -----------------------------
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow
lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# -----------------------------
# Detect features in first frame
# -----------------------------
first_color, first_gray = images[0]

p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

prev_gray = first_gray
prev_pts = p0

# draw first frame
frame_vis = first_color.copy()

for pt in p0:
    x, y = pt.ravel()
    cv2.circle(frame_vis, (int(x), int(y)), 4, (0,255,0), -1)

cv2.imwrite("output/frame_000.png", frame_vis)

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

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if(i == 1):
        foe = compute_foe(good_old, good_new)

    print("FOE:", foe)

    frame_vis = color.copy()

    # draw motion
    for new, old in zip(good_new, good_old):

        x_new, y_new = new.ravel()
        x_old, y_old = old.ravel()

        # draw point
        cv2.circle(frame_vis, (int(x_new), int(y_new)), 4, (0,255,0), -1)

        # draw motion vector
        cv2.line(frame_vis,
                 (int(x_old), int(y_old)),
                 (int(x_new), int(y_new)),
                 (0,0,255), 1)

    # save frame
    cv2.circle(frame_vis, (int(foe[0]), int(foe[1])), 8, (255,0,0), -1)

    cv2.imwrite(f"output/frame_{i:03d}.png", frame_vis)

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1,1,2)

# --------------------
## now get distances ##
# --------------------
import matplotlib.pyplot as plt

r_values = []
frame_ids = []

# approximate focus of expansion as image center
h, w = images[0][1].shape
center = np.array([w/2, h/2])

prev_gray = images[0][1]
prev_pts = p0

for i in range(1, len(images)):

    color, gray = images[i]

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        **lk_params
    )

    good_new = next_pts[status == 1]

    # compute radial distances
    r = np.linalg.norm(good_new - foe, axis=1)

    r_values.append(np.mean(r))
    frame_ids.append(i)

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1,1,2)

r_values = np.array(r_values)
frame_ids = np.array(frame_ids)

# radial velocity
dr = np.gradient(r_values)

# TTC per frame
ttc_frames = np.abs(r_values / dr)

# use median for robustness
ttc_estimate = np.median(ttc_frames)

print("r values:", r_values)
print("dr values:", dr)
print("Estimated Time To Impact (frames):", ttc_estimate)

# linear fit of r vs frame
coeff = np.polyfit(frame_ids, r_values, 1)

fit_line = np.poly1d(coeff)

# find where r = 0
impact_frame = -coeff[1] / coeff[0]

print("Impact predicted at frame:", impact_frame)

plt.figure()

plt.plot(frame_ids, r_values, 'o', label="Measured radial distance")
plt.plot(frame_ids, fit_line(frame_ids), '-', label="Linear fit")

plt.axvline(impact_frame, linestyle='--', label="Predicted impact frame")

plt.xlabel("Frame")
plt.ylabel("Average radial feature distance (pixels)")
plt.title("Time To Impact Estimation")
plt.legend()

plt.savefig("time_to_impact.png")