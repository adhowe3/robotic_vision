import numpy as np
import matplotlib.pyplot as plt

# given
frames = 18
motion_per_frame = 15.25  # mm

# impact frame from Task 1
impact_frame = 61.9

frame_numbers = np.arange(frames)

# distance remaining until impact
distances = (impact_frame - frame_numbers) * motion_per_frame

print("Distance to object at first frame:", distances[0], "mm")
print("Distance to object at last frame:", distances[-1], "mm")

# plot
plt.figure()

plt.plot(frame_numbers, distances, marker='o')

plt.xlabel("Frame Number")
plt.ylabel("Distance to Object (mm)")
plt.title("Object Distance vs Frame (Known Velocity)")
plt.grid()

plt.axvline(impact_frame, linestyle='--', color='red', label="Predicted Impact")
plt.legend()

plt.savefig("time_to_impact_2.png")