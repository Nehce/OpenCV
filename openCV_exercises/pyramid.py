import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("52870588_p0.jpg")

lr = cv2.pyrDown(img)
lr2 = cv2.pyrDown(lr)
hr = cv2.pyrUp(lr2)

cv2.imshow('Original IMG', img)
cv2.imshow('pyrDOWN', lr)
cv2.imshow('pyrDOWN2', lr2)
cv2.imshow('pyrUP', hr)

cv2.waitKey(0)
cv2.destroyAllWindows()
