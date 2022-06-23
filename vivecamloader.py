import cv2
import numpy as np


#find camera based on the name
def find_camera(name):
    for i in range(0, cv2.VideoCapture.listDevices()):
        if cv2.VideoCapture.get(i).get(cv2.CAP_PROP_NAME) == name:
            return i
    return -1

#load the camera as a 8 bit float
cap = cv2.VideoCapture(find_camera("USB Camera"))
#make 8 bit float
