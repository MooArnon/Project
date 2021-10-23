"""
    This is OpenCv project that was designed for check the settlement of bridge. 
    By using OpenCv and ArUco, the displacement of ArUco will be measured to calculate settlement.
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
import matplotlib as plt

cap = cv2.VideoCapture(0) # Use the build in cam

#* Run the program
while cap.isOpened():
    # Receive value from webcam 
    check, frame = cap.read()
    # Resize cam
    frame = cv2.resize(frame, (900, 1500))
    # Show webcam 
    cv2.imshow("Processing", frame) 
    
    
    
    
    # Exit when press q
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
#* Return ram
cap.release()
cv2.destroyAllWindow()



print("############################################## NO ERRORS ########################################")