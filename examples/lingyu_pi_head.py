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

_ROLL_GRAPH_NAME = '/home/pi/AIY-projects-python/lingyu_tuf/deepgaze/examples/lingyu_graph.binaryproto'





def model_roll():
    return ModelDescriptor(name='roll_inference',input_shape=(1, 64, 64, 3),input_normalizer=(128, 128),compute_graph=utils.load_compute_graph(_ROLL_GRAPH_NAME))
# image_resized = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)

def main():
    num_frames = 90
    with PiCamera() as camera:
        camera.sensor_mode = 4
        camera.resolution = (1640,1232)
        camera.framerate = 30
        camera.start_preview()
        with CameraInference(model_roll()) as inference:
            start = time.time()
            for i, result in enumerate(inference.run()):
                if i == num_frames:
                    break
                print(result)#roll_vector = np.multiply(roll_raw, 25.0) #cnn-out is in range [-1, +1] --> [-45, + 45]
                
        camera.stop_preview()
    end = time.time()
    seconds = end- start
    print("time taken: {0}".format(seconds))
    fps = num_frames/seconds
    print("frames from testing:{0}".format(fps))        
    


if __name__ == "__main__":
    main()