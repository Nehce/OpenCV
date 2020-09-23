import cv2
import numpy as np

pm1 = cv2.imread('001Bulbasaur.png')
pm2 = cv2.imread('002Ivysaur.png')

fusion = np.hstack((pm1[:, :300], pm2[:, 300:]))

# 高斯金字塔
pm1_copy = pm1.copy()
gp_pm1 = [pm1_copy]

for i in range(6):
    pm1_copy = cv2.pyrDown(pm1_copy)
    gp_pm1.append(pm1_copy)

pm2_copy = pm2.copy()
gp_pm2 = [pm2_copy]
for i in range(6):
    pm2_copy = cv2.pyrDown(pm2_copy)
    gp_pm2.append(pm2_copy)

# laplacian pyramid
pm1_copy = gp_pm1[5]
lp_pm1 = [pm1_copy]
for i in range(5, 0, -1):
    shape = gp_pm1[i-1].shape[:2]
    gaussian_expanded = cv2.pyrUp(gp_pm1[i], dstsize=shape)
    print(gp_pm1[i-1].shape, gaussian_expanded.shape)
    laplacian = cv2.subtract(gp_pm1[i-1], gaussian_expanded)
    lp_pm1.append(laplacian)

pm2_copy = gp_pm2[5]
lp_pm2 = [pm2_copy]
for i in range(5, 0, -1):
    shape = gp_pm2[i - 1].shape[:2]
    gaussian_expanded = cv2.pyrUp(gp_pm2[i], dstsize=shape)
    laplacian = cv2.subtract(gp_pm2[i-1], gaussian_expanded)
    lp_pm2.append(laplacian)

pm_pyramid = []
n = 0
# 每个拉普拉斯层进行融合
for pm1_lap, pm2_lap in zip(lp_pm1, lp_pm2):
    n += 1
    cols, rows, ch = pm1_lap.shape
    lp = np.hstack((pm1_lap[:, :int(cols/2)], pm2_lap[:, int(cols/2):]))
    pm_pyramid.append(lp)
it = iter(range(6))
for i in pm_pyramid:

    cv2.imshow(f'i{next(it)}', i)
pm_reconstruct = pm_pyramid[0]
for i in range(1, 6):
    shape = pm_pyramid[i].shape[0:2]
    pm_reconstruct = cv2.pyrUp(pm_reconstruct, dstsize=shape)
    pm_reconstruct = cv2.add(pm_pyramid[i], pm_reconstruct)

cv2.imshow('pm1', pm1)
cv2.imshow('pm2', pm2)
cv2.imshow('fusion', fusion)
cv2.imshow('lap_fusion', pm_reconstruct)

cv2.waitKey(0)
cv2.destroyAllWindows()
