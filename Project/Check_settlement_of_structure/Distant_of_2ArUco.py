import math
import pandas as pd
import cv2
import cv2.aruco as aruco
import numpy as np

#* Paremetres
# From calibration code
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])

firstMarkerID = 0
secondMarkerID = 1
marker_size = 50 # [mm]
tvecsData = []
distance = []
FrameSize = (1080, 720)
cap = cv2.VideoCapture(0)
x1 = 0
x2 = 0
y1 = 0
y2 = 0  

#* Distant  
def calculateDistance(x1,y1,x2,y2):  # Calculate distant between 2 ArUco
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)      # Formula
     return dist    # Return result

#* Process
def process(CameraMatrix, CameraDistotion):
    while cap.isOpened():
        ret, frame =  cap.read()
        frame = cv2.resize(frame, FrameSize)    # Resize camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gray parameter, use for detect ArUco
        arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)    # Get 5x5 ArUco
        parameter = aruco.DetectorParameters_create()   #Create detect parametres
        (corners, ids, rejected_img_points) = aruco.detectMarkers(gray, arucodict,  #Detect corners and id of marker
                                                                parameters=parameter,
                                                                cameraMatrix=CameraMatrix,
                                                                distCoeff=CameraDistotion)
        if np.all(ids is not None): # Detect marker
            for i in range(0, len(ids)): # Detect multi marker
                rvec, tvec, marker_point = aruco.estimatePoseSingleMarkers(corners[i], marker_size,  # get tvec from marker
                                                                        CameraMatrix, CameraDistotion)
                if ids[i] == firstMarkerID:     # Detect 1st marker
                    x1 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y1 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    # Print position 1st each marker
                    cv2.putText(frame, 'x1  '+str(x1), (200,400), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    cv2.putText(frame, 'y1  '+str(y1), (50,400), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

                elif ids[i] == secondMarkerID:  # Detect 2nd marker
                    x2 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y2 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    # Print position of 2nd marker
                    cv2.putText(frame, 'x2  '+str(x2), (200,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    cv2.putText(frame, 'y2  '+str(y2), (50,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

            #######* Distance calculation process *########
            if len(ids) > 1 and x1 is not None and x2 is not None:
                dist = round(calculateDistance(x1, y1, x2, y2), 2) # , [millimetres]
                cv2.putText(frame, 'Distant:  '+str(dist), (800,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                aruco.drawDetectedMarkers(frame, corners)
            
        cv2.imshow('Processing', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Exit
                break
        elif key == ord('f'):
            print('FUCK YOU!!!!!!!!!!!!!!!!!!!!')   # Easter Egg
    cap.release()
    cv2.destroyAllWindows()

process(camera_matrix, camera_distotion)