import numpy as np
import cv2
from numpy.lib.type_check import _imag_dispatcher
import cv2.aruco as aruco
import pandas as pd
import glob
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])

imges = glob.glob('/Users/moomacprom1/Data_science/Code/GitHub/Project/Check_settlement_of_structure/MaximumDistance/Data/*.jpg')
imges = sorted(imges)


marker_size = 50 # ,[mm]
def processing(cameraMatrice, cameraDistotion):
    x = 0
    for image in imges:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_point = aruco.detectMarkers(gray, arucodict,
                                                                parameters=parameters,
                                                                cameraMatrix=camera_matrix,
                                                                distCoeff=camera_distotion)
        
        if np.all(ids is not None):
            for i in range(0, len(ids)):
                rvec, tvec, markerPoint = aruco.estimatePoseSingleMarkers(corners[i], marker_size, 
                                                                        cameraMatrice, cameraDistotion)
                distant = round(tvec[0][0][2], 2)
                aruco.drawDetectedMarkers(img, corners)
                aruco.drawAxis(img, camera_matrix, camera_distotion, rvec, tvec, 20)
        else:
            break


        
        cv2.imshow("Processing", img)
        cv2.waitKey(1500)

        x += 1
        print('Processing image ', x, 'with distant ', distant,'millimetres')

processing(camera_matrix, camera_distotion)

