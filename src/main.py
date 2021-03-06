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
b_hls = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2HLS)
b_h, b_l, b_s = b_hls[:, :, 0], b_hls[:, :, 1], b_hls[:, :, 2]
b_cp = np.copy(b_l)


def ir_analysis(filename):
    global cp
    img = cv2.imread(filename)
    # # Add black spots
    for i in range(0, 20):
        x = np.random.randint(low=0, high=300)
        y = np.random.randint(low=0, high=220)
        p0 = x, y
        p1 = x+5, y+5
        cv2.rectangle(img, p0, p1, (5, 5, 5), cv2.FILLED)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = hls_img[:, :, 0], hls_img[:, :, 1], hls_img[:, :, 2]

    cp = np.copy(l)
    hot = cold = np.zeros(cp.shape)
    hotness = coldness = 0
    for row in range(cp.shape[0]):
        for col in range(cp.shape[1]):
            pixel_delta = cp[row][col].astype(float) - b_cp[row][col].astype(float)
            if pixel_delta == 0:
                pass
            elif pixel_delta > 0:
                hot[row][col] = pixel_delta
                hotness += pixel_delta
            else:
                cold[row][col] = pixel_delta
                coldness += pixel_delta
    return img, cp, hot, cold, hotness, coldness


def make_plot():
    # Plot data
    rows, cols = len(image_paths), 4
    for count, path in enumerate(image_paths):
        index = 4*count
        img, lightness, hot, cold, total_heat, total_cool = ir_analysis(path)
        hot = np.ma.masked_where(hot <= 0, hot)
        cold = np.ma.masked_where(cold >= 0, cold)
        print(f"Total heat: {total_heat}")
        print(f"Total cool: {total_cool}")
        # Add baseline
        plt.subplot(rows, cols, index+1)
        plt.title(f"Baseline")
        plt.imshow(cv2.cvtColor(baseline_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # Add image
        plt.subplot(rows, cols, index+2)
#        plt.title(f"{image_name}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        # Add hot
        plt.subplot(rows, cols, index+3)
        plt.imshow(lightness, cmap="gray")
        plt.imshow(hot, cmap='hot')
        plt.axis('off')
        # Add cold
        plt.subplot(rows, cols, index+4)
        plt.title(f"Cold")
        plt.imshow(lightness, cmap="gray")
        plt.imshow(cold, cmap='cool')
        plt.axis('off')
    plt.show()


