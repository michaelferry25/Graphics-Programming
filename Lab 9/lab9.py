import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reads the image
img = cv2.imread("ATU1.jpg")

rows = 3
cols = 3

# Converts the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgHarris = gray_image.copy()

blockSize = 2
aperture_size = 3
k = 0.04

dst = cv2.cornerHarris(imgHarris, blockSize, aperture_size, k)

threshold = 0.04; 
B = 0
G = 0
R = 0
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

# Checks if image is loaded correctly
if img is None:
    print("Error: Could not load image.")
    exit()

# Original Image
plt.subplot(cols, cols,1)
plt.imshow(cv2.cvtColor(img, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])

# Gray Scale Image
plt.subplot(cols, cols,2)
plt.imshow(gray_image, cmap = 'gray')
plt.title("Gray Scale")
plt.xticks([])
plt.yticks([])

# Harris Corners
plt.subplot(cols, cols,3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.xticks([])
plt.yticks([])

plt.subplot(cols, cols,4)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
