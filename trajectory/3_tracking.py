import cv2
import numpy as np
import os

# keep the start-end index images, remove the rest from the dataset 
def keep_image_range(base_path, start, end):
    for side in ["L", "R"]:
        folder = os.path.join(base_path, side)

        for file in os.listdir(folder):
            if file.endswith(".png"):
                idx = int(file.split(".")[0])

                if idx < start or idx > end:
                    os.remove(os.path.join(folder, file))


def rename_files(base_path):
    """
    Rename all PNG files in L/ and R/ subfolders of base_path
    to 0.png, 1.png, 2.png, ... based on sorted order.
    """
    for side in ["L", "R"]:
        folder = os.path.join(base_path, side)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))

        for idx, f in enumerate(files):
            old_path = os.path.join(folder, f)
            new_path = os.path.join(folder, f"{idx}.png")
            os.rename(old_path, new_path)

################# main ######################
if __name__ == "__main__":
    rename_files("ball_images")