import numpy as np
import pandas as pd
import cv2
import cv2.aruco as aruco

#* Inverse second vector
def InverseVector(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

#* Relative position
def RelativePosition(tvec1, rvec1, tvec2, rvec2):
    tvec1, rvec1 = tvec1.reshape((3,1)), rvec1.reshape((3,1))
    tvec2, rvec2 = tvec2.reshape((3,1)), rvec2.reshape((3,1))
    invtvec, invrvec = InverseVector(tvec2, rvec2)
    info = cv2.composeRT(rvec1, tvec1, invrvec, invtvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3,1))
    composedTvec = composedTvec.reshape((3,1))
    return composedRvec, composedTvec

#* Camera caribrated parameters
#! Not true value, need to calibrate again
camera_matrix = np.array([[1.019099074177694320e+03, 0.00000000e+00, 6.557727729771451095e+02,], 
                       [0.00000000e+00, 1.011927236550148677e+03, 3.816077913964442700e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[2.576784605153304430e-01,-1.300640184051879311e+00, -4.285777480424158084e-03,
                              -2.507657388926626523e-03, 2.307018624520866812e+00]])

def working(camera_matrix, camera_distotion):
    #* OpenCV section
    # Parameter
    PointCircle = (0,0)
    MarkerTvecList = []
    MarkerRvecList = []
    composedRvec, composedTvec = None, None
    firstMarkerID = 0
    secondMarkerID = 1

    # Read video from build in camera.
    cap = cv2.VideoCapture(0)
    # Open program
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameter = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, arucodict,
                                                                parameters=parameter,
                                                                cameraMatrix=np.float32(camera_matrix),
                                                                distCoeff=np.float32(camera_distotion))
        # Detect all camera
        if np.all(ids is not None):
            del MarkerRvecList[:]
            del MarkerTvecList[:]
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):
                rvec, tvec, markerPoint = aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix, camera_distotion)
                if ids[i] == firstMarkerID:
                    firstRvec = rvec
                    firstTvec = tvec
                elif ids[i] == secondMarkerID:
                    secondRvec = rvec
                    secondTvec = tvec
                (rvec - tvec).any()
                MarkerRvecList.append(rvec)
                MarkerTvecList.append(tvec)
                aruco.drawDetectedMarkers(frame, corners)
            
            if len(ids) > 1 and composedTvec is not None and composedRvec is not None:
                info = cv2.composeRT(composedRvec, composedTvec, secondRvec.T, secondTvec.T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]

                imgpts = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, camera_matrix, camera_distotion)
                aruco.drawAxis(frame, camera_matrix, camera_distotion, TcomposedRvec, TcomposedTvec, 0.01)
                relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                cv2.circle(frame, relativePoint, 2, (255, 255, 0))
        #display
        cv2.imshow('Processing', frame)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
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

