import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./ATU_GRAY.jpg',gray_image)

nrows = 2
ncols = 2
blurredGray = cv2.GaussianBlur(gray_image,(7,7),0)


plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title("GRAY SCALE"), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,3),plt.imshow(blurredGray, cmap = 'gray')
plt.title('Original Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4),plt.imshow(cv2.GaussianBlur(gray_image,(13,13),4), cmap = 'gray')
plt.title("GRAY SCALE BLUR 13x13"), plt.xticks([]), plt.yticks([])

plt.show()