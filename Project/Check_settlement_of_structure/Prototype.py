"""
    This is OpenCv project that was designed for check the settlement of bridge. 
    By using OpenCv and ArUco, the displacement of ArUco will be measured to calculate settlement.
"""
from os import name
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import matplotlib as plt
import math

cap = cv2.VideoCapture(0) # Use the build in cam

bridgeNO = 1
marker_size = 5 # [cm]

#* Formulation
# invers vector BC
def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec
# find relative position of two points
def relativePosition(rvec1, tvec1, rvec2, tvec2):
    ## Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    ## Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

#* Calibration factor
CalibrationPath =""
camera_matrix = np.array([[1.019099074177694320e+03, 0.00000000e+00, 6.557727729771451095e+02,], 
                       [0.00000000e+00, 1.011927236550148677e+03, 3.816077913964442700e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[2.576784605153304430e-01,-1.300640184051879311e+00, -4.285777480424158084e-03,-2.507657388926626523e-03, 2.307018624520866812e+00]])
#* Run the program
while cap.isOpened():
    # Receive value from webcam 
    check, frame = cap.read()
    # Resize cam
    frame = cv2.resize(frame, (1280, 720))
    # Add gray frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    # Detect ArUco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejected = aruco.detectMarkers(gray, 
                                                 aruco_dict,
                                                 parameters=parameters,
                                                 cameraMatrix=np.float32(camera_matrix),
                                                 distCoeff=np.float32(camera_distotion))
    if np.all(ids is not None): 
        for i in range(0, len(ids)):
            ret = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, camera_distotion)
            # unpack output
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
            # Draw detected marker, put axis
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, camera_matrix, camera_distotion, rvec, tvec, 10) 
    
    # Text section
    ##name
    name_of_project = "Bridge number " + str(bridgeNO)
    cv2.putText(frame, name_of_project,
                (640,50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2) #BGR
                

    # Show webcam 
    cv2.imshow("Processing", frame) 
    
    # Exit when press q
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
#* Return ram
cap.release()
cv2.destroyAllWindows()  



print("############################################## NO ERRORS ########################################")