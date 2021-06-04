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
b_gray = np.copy(baseline_gray)


def intensity(filename, pixel_temperature_threshold):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cp = np.copy(img_gray)
    hot = cold = 0
    for row in range(cp.shape[0]):
        for col in range(cp.shape[1]):
            # Get hot increase
            pixel_delta = cp[row][col] - b_gray[row][col]
            print(pixel_delta)
            if pixel_delta > 0:
                hot += pixel_delta
            elif pixel_delta < 0:
                cold[row][col] = pixel_delta

            else:
                pass
        return


for count, path in enumerate(image_paths):
    print(path)
    intensity(path, 148)
