import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("52870588_p0.jpg", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
sobel_X = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobel_X = np.uint8(np.absolute(sobel_X))
sobel_Y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel_Y = np.uint8(np.absolute(sobel_Y))
sobel_XY = cv2.bitwise_or(sobel_X, sobel_Y)
canny = cv2.Canny(img, 100, 200)

titles = ['image', 'Laplacian', 'sobel_X', 'sobel_Y', 'sobel_XY', 'canny']
images = [img, lap, sobel_X, sobel_Y, sobel_XY, canny]
for i in range(len(titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

