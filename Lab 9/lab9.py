import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO, solutions

img = cv2.imread("ATU1.jpg")


rows = 3 # How much rows in our popup
cols = 3 # How much columns in our popup

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgHarris = img.copy()
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)
threshold = 0.04; 
B = 0
G = 0
R = 255

for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

imgShiTomasi = img.copy()

maxCorners = 100
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(gray_image,maxCorners,qualityLevel,minDistance)
corners = np.int8(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(B, G, R),-1)


orbImage = cv2.imread('ATU1.jpg', cv2.IMREAD_GRAYSCALE)
 
# Initiate ORB detector
orb = cv2.ORB_create()
 
# find the keypoints with ORB
kp = orb.detect(orbImage,None)
 
# compute the descriptors with ORB
kp, des = orb.compute(orbImage, kp)


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

#Shi Tomasi
plt.subplot(cols, cols,4)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title("Shi Tomasi")
plt.xticks([])
plt.yticks([])

# Orb Image
plt.subplot(cols, cols,5)
plt.imshow(cv2.drawKeypoints(orbImage, kp, None, color=(0,255,0), flags=0))
plt.title("ORB Image")
plt.xticks([])
plt.yticks([])

plt.show()

def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    counter = solutions.ObjectCounter(show=True, model=model_path, classes=classes_to_count)
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
count_specific_classes("100m.mp4", "output_specific_classes.avi", "yolo11n.pt", [0])