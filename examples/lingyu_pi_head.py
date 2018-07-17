from __future__ import division
import numpy as np
import os.path
import imp #to check for missing modules
import math
from aiy.vision.inference import ModelDescriptor
from aiy.vision.models import utils
from aiy.vision.models import face_detection
import cv2
import time
from PIL import Image
from aiy.vision.inference import ImageInference

_ROLL_GRAPH_NAME = 'lingyu_roll-graph.binaryproto'
_PITCH_GRAPH_NAME = 'lingyu_pitch-graph.binaryproto'
_YAW_GRAPH_NAME = 'lingyu_yaw-graph.binaryproto'

# #Check if dlib is installed
# try:
#     imp.find_module('dlib')
#     IS_DLIB_INSTALLED = True
#     import dlib
#     print('[DEEPGAZE] head_pose_estimation.py: the dlib library is installed.')
# except ImportError:
#     IS_DLIB_INSTALLED = False
#     print('[DEEPGAZE] head_pose_estimation.py: the dlib library is not installed.')

#Enbale if you need printing utilities
DEBUG = True



def model_roll():
    return ModelDescriptor(name='roll_inference',input_shape=(1, 0, 0, 3),input_normalizer=(0, 0),compute_graph=utils.load_compute_graph(_ROLL_GRAPH_NAME))
def model_pitch():
    return ModelDescriptor(name='pitch_inference',input_shape=(1, 0, 0, 3),input_normalizer=(0, 0),compute_graph=utils.load_compute_graph(_PITCH_GRAPH_NAME))
def model_yaw():
    return ModelDescriptor(name='yaw_inference',input_shape=(1, 0, 0, 3),input_normalizer=(0, 0),compute_graph=utils.load_compute_graph(_YAW_GRAPH_NAME))
def model_face():
    return ModelDescriptor(name='yaw_inference',input_shape=(1, 0, 0, 3),input_normalizer=(0, 0),compute_graph=utils.load_compute_graph(_YAW_GRAPH_NAME))

def return_face(image, radians=False):
    with ImageInference(face_detection.model()) as inference:
            for i, face in enumerate(face_detection.get_faces(inference.run(image))):
                print('Face #%d: %s' % (i, str(face)))
                x, y, width, height = face.bounding_box
            return None
def return_roll(image, radians=False):
    with ImageInference(model_roll()) as roll_inference:
        for i, roll_raw in enumerate(roll_inference.run(image)):
            roll_vector = np.multiply(roll_raw, 25.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
            if(radians==True): return np.multiply(roll_vector, np.pi/180.0) #to radians
            else: return roll_vector
def return_pitch(image, radians=False):
    with ImageInference(model_pitch()) as pitch_inference:
        for i, pitch_raw in enumerate(pitch_inference.run(image)):
            pitch_vector = np.multiply(pitch_raw, 45.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
            if(radians==True): return np.multiply(pitch_vector, np.pi/180.0) #to radians
            else: return pitch_vector 
    
def return_yaw(image, radians=False):
    with ImageInference(model_yaw()) as yaw_inference:
        # Add resize function to put the image into 64*64*3
        # image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
        # image_normalised = np.add(image_resized, -127) #normalisation of the input
        # feed_dict = {self.tf_yaw_input_vector : image_normalised}
        # yaw_raw = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)
        # if(h != w or w<64 or h<64):
        #     if h != w :
        #         raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height must equal Width. Height=%d,Width=%d'%(h,w))
        #     else:
        #         raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input has wrong shape. Height and Width must be >= 64 pixel')
        # #wrong number of channels
        # if(d!=3):
        #     raise ValueError('[DEEPGAZE] CnnHeadPoseEstimator(return_yaw): the image given as input does not have 3 channels, this function accepts only colour images.')
        for i, yaw_raw in enumerate(yaw_inference.run(image)):
            yaw_vector = np.multiply(yaw_raw, 100.0)
            if(radians==True): return np.multiply(yaw_vector, np.pi/180.0) #to radians
            else: return yaw_vector
def main():
    num_frames = 90
    with CameraInference(face_detection.model()) as inference:
        start = time.time()
        for i, result in enumerate(inference.run()):
            if i == num_frames:
                break
            faces = face_detection.get_faces(result)
    end = time.time()
    seconds = end- start
    print("time taken: {0}".format(seconds))
    fps = num_frames/seconds
    print("frames from testing:{0}".format(fps))        
##            annotator.clear()
##            for face in faces:
##                annotator.bounding_box(transform(face.bounding_box), fill=0)
##            annotator.update()
##            print('Iteration #%d: num_faces=%d' % (i, len(faces)))

##    #Defining the video capture object
##    video_capture = cv2.VideoCapture(0)
##    video_capture.set(3,256)
##    video_capture.set(4,256)
##    
##    if(video_capture.isOpened() == False):
##        print("Error: the resource is busy or unvailable")
##    else:
##        print("The video source has been opened correctly...")
##
##    #Create the main window and move it
##    cv2.namedWindow('Video')
##    cv2.moveWindow('Video', 20, 20)
##
##
##    num_frames = 90
##    print("capture {0} frames".format(num_frames))
##    start = time.time()
##    for i in range(0, num_frames):
##        ret, frame = video_capture.read()
##        print(i)
##        sq_frame = Image.fromarray(frame)
##        roll = return_face(sq_frame)
####        print(str(roll[0,0,0]))
####        lbp_face_cascade = cv2.CascadeClassifier('./etc/xml/lbpcascade_frontalface.xml')
####
####        #call our function to detect faces
####        faces_detected_img = detect_faces(lbp_face_cascade, frame)
##    end = time.time()
##    seconds = end- start
##    print("time taken: {0}".format(seconds))
##    fps = num_frames/seconds
##    print("frames from testing:{0}".format(fps))
##    #Release the camera
##    video_capture.release()
##    print("Bye...")


if __name__ == "__main__":
    main()