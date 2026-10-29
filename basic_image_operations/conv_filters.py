import numpy as np
import cv2 as cv

K1 = np.array([[1,2,1],
               [2,4,2],
               [1,2,1]])/16
K2_1 = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]])
K2_2 = np.array([[-1,-2,1],
               [0,0,0],
               [1,2,1]])
K3 = np.array([[0,-1,0],
               [-1,5,-1],
               [0,-1,0]])
K4 = np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])/9
img = cv.imread('../data/in/squirre.jpg', cv.IMREAD_GRAYSCALE)
#img = cv.imread('../data/in/forrest.jpg', cv.IMREAD_GRAYSCALE)
print(img.shape)
im2 = cv.filter2D(src=img, ddepth=-1, kernel=K2_1)
im3 = cv.filter2D(src=img, ddepth=-1, kernel=K2_2)

cv.imshow("orig",img)
cv.imshow('K2_1', im2)
cv.imshow('K2_2', im3)
cv.waitKey(0)
cv.destroyAllWindows()
minVal = 100
maxVal = 200
edges = cv.Canny(img,minVal,maxVal)
cv.imshow('Canny ({},{})'.format(minVal,maxVal), edges)
minVal = 180
maxVal = 240
edges = cv.Canny(img,minVal,maxVal)
cv.imshow('Canny ({},{})'.format(minVal,maxVal), edges)
cv.waitKey(0)
cv.destroyAllWindows()