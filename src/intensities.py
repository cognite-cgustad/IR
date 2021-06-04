import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

image_paths = []
for root, dirs, files in os.walk("./../img", topdown=False):
    for name in files:
        image_paths.append(os.path.join(root, name))
image_paths.sort()

# Pick baseline image as inital image
baseline_path = image_paths.pop(0)
baseline_image = cv2.imread(baseline_path)
baseline_gray = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)
bcp = np.copy(baseline_gray.astype(float))


def make_plot():
    # Plot data
    rows, cols = len(image_paths), 4
    for count, path in enumerate(image_paths):
        image_name = path.split('/')[-1]
        # Get image and compute delta
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cp = np.copy(img_gray.astype(float))
        delta = cp - bcp
        # Split array on sign
        heat = np.maximum(delta, 0)
        cold = np.maximum(-delta, 0)
        # Mask arrays
        heat = np.ma.masked_where(heat <= 0, heat)
        cold = np.ma.masked_where(cold <= 0, cold)
        num_pixels_increase = heat.count()
        num_pixels_decrease = cold.count()
        total_temp_increase = np.sum(heat)
        total_temp_decrease = np.sum(cold)
        print(f"In image {image_name}")
        print(f"Found {num_pixels_increase} pixels with temprature increase. Total increase found to be {total_temp_increase}")
        print(f"Found {num_pixels_decrease} pixels with intensity decrease . Total decrease found to be {total_temp_decrease}")

        # Plot data
        index = 4*count
        # Add baseline
        plt.subplot(rows, cols, index+1)
        plt.title(f"Baseline")
        plt.imshow(cv2.cvtColor(baseline_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # Add image
        plt.subplot(rows, cols, index+2)
        plt.title(f"{image_name}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # Add hot
        plt.subplot(rows, cols, index+3)
        plt.imshow(cp, cmap="gray")
        plt.imshow(heat, cmap='hot')
        plt.axis('off')
        # Add cold
        plt.subplot(rows, cols, index+4)
        plt.title(f"Cold")
        plt.imshow(cp, cmap="gray")
        plt.imshow(cold, cmap='cool')
        plt.axis('off')
    plt.show()

