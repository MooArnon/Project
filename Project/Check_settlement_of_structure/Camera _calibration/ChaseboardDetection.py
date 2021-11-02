import numpy as np 
import pandas as pd
import cv2 as cv
import glob

chessboardSize = (7,10)
frameSize = (1080,720)



objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 40
objp = objp * size_of_chessboard_squares_mm

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image = '/Users/moomacprom1/Data_science/Code/GitHub/Project/Check_settlement_of_structure/Camera _calibration/Data/img1.jpg'

img = cv.imread(image)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    
if ret == True:

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)

        
    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
cv.imshow('img', img)
cv.waitKey(0)
print("########################---No Errors---############################")


cv.destroyAllWindows()


