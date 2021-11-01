import numpy as np
import pandas as pd
import cv2
import cv2.aruco as aruco
"""
    Objective: To find the relative distance between marker
    Conditions: Using 2 ArUco marker, first as a reference point and second is settlement measurement.
    By: Arnon Phongsiang ==email== oomarnon.000@gmail.com
        Suppawich Pinyo  ==email== suppawich.pinyo@mail.kmutt.ac.th
         Nichapon Akkarachaidecho ==email== nichapon.a@mail.kmutt.ac.th

    Concept:

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
    return invRvec, invTvec

#* Relative position
def RelativePosition(rvec1, tvec1,rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3,1)), tvec1.reshape((3,1))
    rvec2, tvec2 =  rvec2.reshape((3,1)), tvec2.reshape((3,1)),
    invrvec, invtvec = InverseVector(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invrvec, invtvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3,1))
    composedTvec = composedTvec.reshape((3,1))
    return composedRvec, composedTvec

#* Camera caribrated parameters
#! Not true value, need to calibrate again.
camera_matrix = np.array([[1.07729557e+03, 0.00000000e+00, 7.40231371e+02,], 
                       [0.00000000e+00, 1.07316875e+03, 5.51345664e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 1.38914028e-01, -3.23710598e+00, -1.31295901e-03,  1.77727072e-03, 3.90343997e+01]])

#* OpenCV section
def working(camera_matrix, camera_distotion):
    # Parameter
    MarkerTvecList = []
    MarkerRvecList = []
    composedRvec, composedTvec = None, None
    firstMarkerID = 0
    secondMarkerID = 1
    marker_size = 0.05

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
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):
                rvec, tvec, markerPoint = aruco.estimatePoseSingleMarkers(corners[i], marker_size, 
                                                                        camera_matrix, camera_distotion)
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

                # frame = draw(frame, corners[0], imgpts)
                aruco.drawAxis(frame, camera_matrix, camera_distotion, TcomposedRvec, TcomposedTvec, 0.03)  # Draw Axis

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

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



working(camera_matrix,camera_distotion)

