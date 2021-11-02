import numpy as np 
import pandas as pd
import cv2
import cv2.aruco as aruco

marker_size = 100 # [mm]

#* Calibration coefficient
CalibrationPath =""
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])
cap = cv2.VideoCapture(0)

#* Run program
while cap.isOpened():
    check, frame = cap.read()
    # resize camera
    frame = cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #* ArUco
    # detect AruCo Marker, 5*5 [cm]
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    # marker detection
    parameters = aruco.DetectorParameters_create()
    # find all markers in video
    corners, ids, rejected = aruco.detectMarkers(gray, 
                                                 aruco_dict,
                                                 parameters=parameters,
                                                 cameraMatrix=np.float32(camera_matrix),
                                                 distCoeff=np.float32(camera_distotion))
    # Detect all camera
    if np.all(ids is not None): 
        for i in range(0, len(ids)):
            ret = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, camera_distotion)
            # unpack output
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
            # Draw detected marker, put axis
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, camera_matrix, camera_distotion, rvec, tvec, 100) 
            
    # Show interface
    cv2.imshow("Processing", frame)
    # Exit when press q
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("########################---No Errors---############################")