import numpy as np
import pandas as pd
import cv2
import cv2.aruco as aruco
"""
    Objective: To find the relative distance between marker
    Conditions: Using 2 ArUco marker, first as a reference point and second is settlement measurement.
    By: Arnon Phongsiang         ==email== oomarnon.000@gmail.com
        Suppawich Pinyo          ==email== suppawich.pinyo@mail.kmutt.ac.th
        Nichapon Akkarachaidecho ==email== nichapon.a@mail.kmutt.ac.th

    Concept: Using cv2.composeRT to combine vector AC and BC. The differences between x-y-z canc be indicated.
             So, the settlemet can be indicated.

            vec(AB)
    [ A ]--------->[ B ]
       \            /
        \          /
         \        /           AB + BC = AC
  vec(AC) \      / vec(BC)    AB = AC - BC
           \    /             AB = AC + (-BC)  # Inverse vector BC and compose them to indicate vector(AB)
            \  /              Then, differences of y-axis will be the settlement.
             VV  
             [c]
"""
# Read video from build in camera.
cap = cv2.VideoCapture(0)

#* Inverse second vector
def InverseVector(rvec, tvec): # inputted rvec is now rotation vector.
    R, _ = cv2.Rodrigues(rvec) # using Rodrigues() to convert rvec, vector, to metric.
    R = np.matrix(R).T         # .T for transpose.
    invTvec = np.dot(-R, np.matrix(tvec)) # Dot product of invese R and tvec
    invRvec, _ = cv2.Rodrigues(R) # convert R back to vector
    return invRvec, invTvec # return inversed vector

#* Relative position
def RelativePosition(rvec1, tvec1,rvec2, tvec2):    # input tvec and tvec from each markers
    rvec1, tvec1 = rvec1.reshape((3,1)), tvec1.reshape((3,1))   # reshape
    rvec2, tvec2 =  rvec2.reshape((3,1)), tvec2.reshape((3,1)), # reshape
    invrvec, invtvec = InverseVector(rvec2, tvec2)  # inverse second vector
    info = cv2.composeRT(rvec1, tvec1, invrvec, invtvec) # conpose vec(AC) and vec(-BC)    
    composedRvec, composedTvec = info[0], info[1]   # get th e result
    composedRvec = composedRvec.reshape((3,1))  # reshape
    composedTvec = composedTvec.reshape((3,1))  # reshape
    return composedRvec, composedTvec   # return resulted vector

#* Camera caribrated parameters
# From calibration code
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])

#* OpenCV section
def working(camera_matrix, camera_distotion):
    # Parameter
    composedRvec, composedTvec = None, None
    firstMarkerID = 0
    secondMarkerID = 1
    marker_size = 50 # [mm]

    # Open program
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameter = aruco.DetectorParameters_create()
        (corners, ids, rejected_img_points) = aruco.detectMarkers(gray, arucodict,
                                                                parameters=parameter,
                                                                cameraMatrix=camera_matrix,
                                                                distCoeff=camera_distotion)
        # Detect all marker
        if np.all(ids is not None):
            for i in range(0, len(ids)): # detect marker
                rvec, tvec, markerPoint = aruco.estimatePoseSingleMarkers(corners[i], marker_size, 
                                                                        camera_matrix, camera_distotion)
                # give parameter
                if ids[i] == firstMarkerID:
                    firstRvec = rvec
                    firstTvec = tvec
                elif ids[i] == secondMarkerID:
                    secondRvec = rvec
                    secondTvec = tvec
                aruco.drawDetectedMarkers(frame, corners)
            
            if len(ids) > 1 and composedTvec is not None and composedRvec is not None:
                info = cv2.composeRT(composedRvec, composedTvec, secondRvec.T, secondTvec.T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]
                
                # position of composed tvec
                cv2.putText(frame, 'Relative position  '+str(TcomposedTvec), (50,400), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

                aruco.drawAxis(frame, camera_matrix, camera_distotion, TcomposedRvec, TcomposedTvec, 10)  # Draw Axis
        


        #display
        cv2.imshow('Processing', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Exit
                break
        elif key == ord('c'): 
            if len(ids) > 1:  # if 2 marker detected reverse second vectors and find the differences, by using RelativePosition
                firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
                secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))
                composedRvec, composedTvec = RelativePosition(firstRvec, firstTvec, secondRvec, secondTvec)
        
        
    cap.release()
    cv2.destroyAllWindows()



working(camera_matrix,camera_distotion)

