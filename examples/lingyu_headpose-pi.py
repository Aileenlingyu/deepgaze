#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# This is an example of head pose estimation with solvePnP.
# It uses the dlib library and openCV
#
import time
import numpy
import cv2
import sys
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection


#If True enables the verbose mode
DEBUG = True 


def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(colored_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return colored_img
def main():

    #Defining the video capture object
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,256)
    video_capture.set(4,256)
    
    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")

    #Create the main window and move it
    cv2.namedWindow('Video')
    cv2.moveWindow('Video', 20, 20)


    num_frames = 120
    print("capture {0} frames".format(num_frames))
    start = time.time()
    for i in range(0, num_frames):
        ret, frame = video_capture.read()
        lbp_face_cascade = cv2.CascadeClassifier('./etc/xml/lbpcascade_frontalface.xml')

        #call our function to detect faces
        faces_detected_img = detect_faces(lbp_face_cascade, frame)
        
        #Showing the frame and waiting
        # for the exit command
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
   
    end = time.time()
    seconds = end- start
    print("time taken: {0}".format(seconds))
    fps = num_frames/seconds
    print("frames from testing:{0}".format(fps))
    #Release the camera
    video_capture.release()
    print("Bye...")



if __name__ == "__main__":
    main()
