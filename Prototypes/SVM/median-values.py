import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"Leaves\Leaves\Sycamore\CIMG0658.JPG")

# Remove white values within a 10 value threshold
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])
white_mask = cv2.inRange(image, lower_white, upper_white)
image_no_white = cv2.bitwise_and(image, image, mask=~white_mask)
print("Image Thresholded")

# Get the green channel
green_channel = image_no_white[:, :, 1]

# Create an array of all three values of every pixel
pixel_values = image_no_white.reshape((-1, 3))

# Find and record the median of the green channel
median_green = np.median(pixel_values[:, 1])
print("Median of the green channel:", median_green)

# Remove the pixels close to the median
threshold = 10
pixel_values_filtered = pixel_values[np.abs(pixel_values[:, 1] - median_green) > threshold]

# Repeat the process twice more
for _ in range(2):
    # Find and record the median of the green channel
    median_green = np.median(pixel_values_filtered[:, 1])
    print("Median of the green channel:", median_green)

    # Remove the pixels close to the median
    pixel_values_filtered = pixel_values_filtered[np.abs(pixel_values_filtered[:, 1] - median_green) > threshold]

# Display the colors
fig, ax = plt.subplots(1, 3, figsize=(8, 2))

for i, colour in enumerate(pixel_values_filtered[:3]):
    colour_patch = np.ones((100, 100, 3), dtype=np.uint8) * colour
    ax[i].imshow(colour_patch.reshape((100, 100, 3)))
    ax[i].axis('off')
    ax[i].set_title(f'Color {i + 1}')

plt.show()