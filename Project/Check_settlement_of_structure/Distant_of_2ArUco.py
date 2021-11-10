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

reffMarkerID = 5
firstMarkerID = 0
secondMarkerID = 1
thirdMarkerID = 2
forthMarkerID = 3
marker_size = 50 # [mm]
tvecsData = []
distance = []
FrameSize = (1080, 720)
cap = cv2.VideoCapture(0)

#* Distant in plane x-y
def calculateDistance_xy(x1,y1,x2,y2):  # Calculate distant between 2 ArUco
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)      # Formula
    return dist    # Return result

#* Distant in plane x-y-z
def calculatedistance_xyz(x1,y1,z1,x2,y2,z2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist

#* Process
def process(CameraMatrix, CameraDistotion, cal):
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
                if ids[i] == reffMarkerID:          # id = 5
                    xr = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    yr = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    zr = round(tvec[0][0][2], 3)    # [[[x, y, z]]]
                    # Print position 1st each marker
                    cv2.putText(frame, 'xr  '+str(xr), (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'yr  '+str(yr), (200,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'zr  '+str(zr), (350,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                
                elif ids[i] == firstMarkerID:       # id = 0
                    x1 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y1 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z1 = round(tvec[0][0][2], 3)    # [[[x, y, z]]]
                    # Print position 1st each marker
                    cv2.putText(frame, 'x1  '+str(x1), (50,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'y1  '+str(y1), (200,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z1  '+str(z1), (350,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == secondMarkerID:      # id = 1
                    x2 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y2 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z2 = round(tvec[0][0][2] ,3)    # [[[x, y, z]]]
                    # Print position of 2nd marker
                    cv2.putText(frame, 'x2  '+str(x2), (50,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y2  '+str(y2), (200,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z2  '+str(z2), (350,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == thirdMarkerID:       # id = 2
                    x3 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y3 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z3 = round(tvec[0][0][2] ,3)    # [[[x, y, z]]]
                    # Print position of 2nd marker
                    cv2.putText(frame, 'x3  '+str(x3), (50,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y3  '+str(y3), (200,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z3  '+str(z3), (350,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == forthMarkerID:       # id = 3
                    x4 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y4 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z4 = round(tvec[0][0][2] ,3)    # [[[x, y, z]]]
                    # Print position of 2nd marker
                    cv2.putText(frame, 'x4  '+str(x4), (50,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y4  '+str(y4), (200,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z4  '+str(z4), (350,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

        
            #######* Distance calculation process *########
            if len(ids) > 1 and xr is not None and x1 is not None and x2 is not None and x3 is not None and x4 is not None: 
                if cal == 'xy':
                    dist = round(calculateDistance_xy(xr, yr, x1, y1), 2) # , [millimetres]
                    cv2.putText(frame, 'Distant, in x-y plane:  '+str(dist), (800,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    aruco.drawDetectedMarkers(frame, corners)
                elif cal == 'xyz':
                    dist = round(calculatedistance_xyz(xr, yr, zr, x1, y1, z1), 2) # , [millimetres]
                    cv2.putText(frame, 'Distant, 3D:  '+str(dist), (800,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    aruco.drawDetectedMarkers(frame, corners)

                #* Coordinate diff
                # Show Topic
                cv2.putText(frame, 'Coordinate Displacement', (50,550), cv2.FONT_HERSHEY_SIMPLEX,0.5,  (0, 150, 0), 2)
                # firstMarker
                yDiff_1 = round(yr-y1, 2) # y diff coordinate
                zDiff_1 = round(zr-z1, 2) # z diff coordinate
                cv2.putText(frame, 'y1 different  '+str(yDiff_1), (50,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
                cv2.putText(frame, 'z1 different  '+str(zDiff_1), (50,700), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
                # secondMarker
                yDiff_2 = round(yr-y2, 2) # y diff coordinate
                zDiff_2 = round(zr-z2, 2) # z diff coordinate
                cv2.putText(frame, 'y2 different  '+str(yDiff_2), (50,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
                cv2.putText(frame, 'z2 different  '+str(zDiff_2), (50,700), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
                # thirdMarker
                yDiff_3 = round(yr-y3, 2) # y diff coordinate
                zDiff_3 = round(zr-z3, 2) # z diff coordinate
                cv2.putText(frame, 'y3 different  '+str(yDiff_3), (50,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
                cv2.putText(frame, 'z3 different  '+str(zDiff_3), (50,700), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
                # forthMarker
                yDiff_4 = round(yr-y4, 2) # y diff coordinate
                zDiff_4 = round(zr-z4, 2) # z diff coordinate
                cv2.putText(frame, 'y4 different  '+str(yDiff_4), (50,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
                cv2.putText(frame, 'z4 different  '+str(zDiff_4), (50,700), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate

                                
        cv2.imshow('Processing', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Exit
            break
        elif key == ord('f'):
            print('FUCK YOU!!!!!!!!!!!!!!!!!!!!')   # Easter Egg
    cap.release()
    cv2.destroyAllWindows()


process(camera_matrix, camera_distotion, cal='xyz')

#? Can we use for loop in distant calculation?
