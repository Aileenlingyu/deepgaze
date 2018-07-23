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
from aiy.vision.inference import CameraInference
from picamera import PiCamera
from lingyu_annotator import Annotator
import matrix_transformation
import picamera
import time
from PIL import Image
from PIL import ImageDraw

_ROLL_GRAPH_NAME = '/home/pi/AIY-projects-python/lingyu_tuf/deepgaze/examples/lingyu_graph.binaryproto'





def model_roll():
    return ModelDescriptor(name='roll_inference',input_shape=(1, 64, 64, 3),input_normalizer=(128, 128),compute_graph=utils.load_compute_graph(_ROLL_GRAPH_NAME))
# image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)

def postprocessing(result):
    
    assert len(result.tensors) == 1
    tensor_name = concat
    tensor = result.tensors[tensor_name]
    scaled_angle, shape = tensor.data, tensor.shape
    roll = scaled_angle[0]*180
    pitch = scaled_angle[1]*90
    yaw = scaled_angle[2]*180
    #roll_vector = np.multiply(roll_raw, 25.0) #cnn-out is in range [-1, +1] --> [-45, + 45]

    return roll, pitch,yaw

def proj(roll, pitch, yaw):
    c_x = 1232 / 2
    c_y = 1640 / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])

    print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
    axis = numpy.float32([[50,0,0],[0,50,0],[0,0,50]])
    camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector
    rot_matrix = euler2rotmat(roll*(np.pi/180),pitch*(np.pi/180),-yaw*(np.pi/180))
    rvec, jacobian = cv2.Rodrigues(rot_matrix)
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
    
    
    location = tuple(imgpts[1].ravel()),tuple(imgpts[2].ravel()),tuple(imgpts[0].ravel())
    return location
#Function used to get the rotation matrix
def euler2rotmat(roll, pitch,yaw):
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

def transform(location,x0,y0):

    rx,ry,px,py,yx,yy = location
    return (scale_x * x0, scale_y * y0, scale_x * rx, scale_x * ry,
            scale_x * x0, scale_y * y0, scale_x * px, scale_x * py,
            scale_x * x0, scale_y * y0, scale_x * yx, scale_x * yy)

def main():
    num_frames = 90
    with PiCamera() as camera:
        camera.sensor_mode = 4
        camera.resolution = (1640,1232)
        camera.framerate = 30
        camera.start_preview()
        annotator = Annotator(camera, dimensions=(320, 240))
        scale_x = 320 / 1640
        scale_y = 240 / 1232
        with CameraInference(model_roll()) as inference:
            start = time.time()
            for i, result in enumerate(inference.run()):
                if i == num_frames:
                    break
                roll, pitch, yaw = postprocessing(result)
                print("roll degree:{0}, pitch degree:{0}, yaw degree:{0}".format(roll,pitch,yaw))
                location = proj(roll,pitch,yaw)
                x0 = 120
                y0 = 160
                annotator.clear()
                annotator.line(transform(location,x0,y0), fill=0)
                annotator.update()
        camera.stop_preview()
    end = time.time()
    seconds = end- start
    print("time taken: {0}".format(seconds))
    fps = num_frames/seconds
    print("frames from testing:{0}".format(fps))        
    


if __name__ == "__main__":
    main()