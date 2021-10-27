import numpy as np 
import pandas as pd
import cv2
import cv2.aruco as aruco

id_to_find = 0
marker_size = 5 # [cm]

#* Calibration coefficient
CalibrationPath =""
camera_matrix = np.array([[1.019099074177694320e+03, 0.00000000e+00, 6.557727729771451095e+02,], 
                       [0.00000000e+00, 1.011927236550148677e+03, 3.816077913964442700e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[2.576784605153304430e-01,-1.300640184051879311e+00,-4.285777480424158084e-03,-2.507657388926626523e-03,2.307018624520866812e+00]])

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
            aruco.drawAxis(frame, camera_matrix, camera_distotion, rvec, tvec, 10) 
            
    # Show interface
    cv2.imshow("Processing", frame)
    # Exit when press q
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
print("########################---No Errors---############################")