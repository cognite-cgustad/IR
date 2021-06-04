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
b_cp = np.copy(baseline_gray)


def ir_analysis(filename, pixel_temperature_threshold):
    img = cv2.imread(filename)
    # # Add black spots
    # for i in range(0, 20):
    #     x = np.random.randint(low = 0, high = 300)
    #     y = np.random.randint(low = 0, high = 220)
    #     p0 = x,y
    #     p1 = x+5,y+5
    #     cv2.rectangle(img,p0,p1,(0,0,0),cv2.FILLED)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cp = np.copy(img_gray)
    hot = np.zeros(cp.shape)
    heat = 0
    cold = np.zeros(cp.shape)
    chill = 0
    for row in range(cp.shape[0]):
        for col in range(cp.shape[1]):
            # Get hot increase
            pixel_delta = cp[row][col] - b_cp[row][col]
            if pixel_delta > 0:
                hot[row][col] = pixel_delta
                print(pixel_delta)
            elif pixel_delta < 0:
                cold[row][col] = pixel_delta
                print(pixel_delta)
            else:
                pass
    return img, cp, hot, cold


# Plot data
rows, cols = len(image_paths), 4

for count, path in enumerate(image_paths):
    index = 4*count
    img, lightness, hot, cold = ir_analysis(path, 148)
    image_name = path.split('/')[-1]
    hot = np.ma.masked_where(hot == 0, hot)
    cold = np.ma.masked_where(cold == 0, cold)
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
    plt.title(f"Heat")
    plt.imshow(lightness,cmap="gray")
    plt.imshow(hot, cmap='hot')
    plt.axis('off')
    # Add cold
    plt.subplot(rows, cols, index+4)
    plt.title(f"Cold")
    plt.imshow(lightness,cmap="gray")
    plt.imshow(cold, cmap='hot')
    plt.axis('off')

plt.show()
#for fname in ['.jpg', '_1w.jpg', '_2w.jpg', '_3w.jpg']:
#    print(fname, pixel_temperature_score('/home/luka/dev/cognite/power-demo-apps/infrared.cache/p9085' + fname, 148))
