import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reads the image
img = cv2.imread("ATU1.jpg")

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



# Displays the original and grayscale images using matplotlib
plt.figure(figsize=(10, 5))

# Original image (BGR converted to RGB for Matplotlib)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Grayscale image
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Optionally save the grayscale image
cv2.imwrite("ATU_GRAY.jpg", gray_image)
print("Grayscale image saved as ATU_GRAY.jpg")