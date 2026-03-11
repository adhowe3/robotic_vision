import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

os.makedirs("output1", exist_ok=True)

# -----------------------------
# Feature detection parameters
# -----------------------------
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01)
)

# -----------------------------
# Compute FOE
# -----------------------------
def compute_foe(p_old, p_new):

    A = []
    b = []

    for (x1,y1),(x2,y2) in zip(p_old,p_new):

        u = x2-x1
        v = y2-y1

        A.append([v,-u])
        b.append(v*x1 - u*y1)

    A = np.array(A)
    b = np.array(b)

    foe,_,_,_ = np.linalg.lstsq(A,b,rcond=None)

    return foe


# -----------------------------
# Load images
# -----------------------------
image_paths = glob.glob("images/T*.jpg")

def frame_number(path):
    return int(re.search(r'\d+',path).group())

image_paths = sorted(image_paths,key=frame_number)

images = []

for p in image_paths:

    img = cv2.imread(p)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    images.append((img,gray))

# -----------------------------
# Detect initial features
# -----------------------------
first_color, first_gray = images[0]

p0 = cv2.goodFeaturesToTrack(first_gray,mask=None,**feature_params)
initial_feature_count = len(p0)

print("Initial features:", initial_feature_count)

prev_gray = first_gray
prev_pts = p0

foe_list = []

r_values = []
frame_ids = []
ttc_values = []

# -----------------------------
# Track features frame to frame
# -----------------------------
for i in range(1,len(images)):

    color, gray = images[i]

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        **lk_params
    )

    good_new = next_pts[status==1]
    good_old = prev_pts[status==1]

    # -----------------------------
    # Remove tiny motions
    # -----------------------------
    motion = np.linalg.norm(good_new-good_old,axis=1)

    valid = motion > 0.5

    good_new = good_new[valid]
    good_old = good_old[valid]

    # -----------------------------
    # Compute FOE
    # -----------------------------
    if i <= 5:
        foe = compute_foe(good_old, good_new)
        foe_list.append(foe)

    if i == 5:
        foe = np.mean(foe_list, axis=0)

    print("Frame",i,"FOE:",foe)

    # -----------------------------
    # Compute radial expansion
    # -----------------------------
    r_old = np.linalg.norm(good_old - foe, axis=1)
    r_new = np.linalg.norm(good_new - foe, axis=1)

    dr = r_new - r_old

    expanding = dr > 0.01

    r_old = r_old[expanding]
    dr = dr[expanding]

    if len(r_old) > 0:

        r_values.append(np.mean(r_old))
        frame_ids.append(i)

        ttc = r_old / dr
        ttc_values.append(np.median(ttc))

        impact_predictions = frame_ids + np.array(ttc_values)

        print("Impact predictions per frame:", impact_predictions)

        predicted_impact_frame = np.median(impact_predictions)

        print("Predicted impact frame:", predicted_impact_frame)

        print("Frame",i,"median TTC:",np.median(ttc))

    # -----------------------------
    # Visualization
    # -----------------------------
    vis = color.copy()

    for new,old in zip(good_new,good_old):

        x_new,y_new = new
        x_old,y_old = old

        cv2.circle(vis,(int(x_new),int(y_new)),3,(0,255,0),-1)

        cv2.line(vis,
                 (int(x_old),int(y_old)),
                 (int(x_new),int(y_new)),
                 (0,0,255),1)

    cv2.circle(vis,(int(foe[0]),int(foe[1])),8,(255,0,0),-1)

    cv2.imwrite(f"output1/frame_{i:03d}.png",vis)

    # -----------------------------
    # Add new features if needed
    # -----------------------------
    if len(good_new) < 0.8 * initial_feature_count:

        print("adding more features...")

        mask = np.ones_like(gray)*255

        for pt in good_new:
            x,y = pt
            cv2.circle(mask,(int(x),int(y)),10,0,-1)

        new_pts = cv2.goodFeaturesToTrack(gray,mask=mask,**feature_params)

        if new_pts is not None:
            good_new = np.vstack((good_new,new_pts.reshape(-1,2)))

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1,1,2)

# -----------------------------
# Final TTC Estimate
# -----------------------------
ttc_estimate = np.median(ttc_values)

remaining_frames = predicted_impact_frame - 17
distance = remaining_frames * 15.25
print("remaining frames: ", remaining_frames)
print("distance: ", distance)

## plot
plt.figure()

plt.plot(frame_ids, ttc_values, marker='o')

plt.xlabel("Frame")
plt.ylabel("Estimated TTC (frames)")
plt.title("Time To Impact Estimate")

plt.grid()

plt.savefig("output1/ttc_vs_frame.png")