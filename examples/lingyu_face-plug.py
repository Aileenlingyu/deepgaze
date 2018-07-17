#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Camera inference face detection demo code.

Runs continuous face detection on the VisionBonnet and prints the number of
detected faces.

Example:
face_detection_camera.py --num_frames 10
"""
import argparse
from inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.annotator import Annotator
from picamera import PiCamera
import picamera
import picamera.array
import numpy as np
import io
from PIL import Image



def main():
    """Face detection camera inference example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_frames',
        '-n',
        type=int,
        dest='num_frames',
        default=-1,
        help='Sets the number of frames to run for, otherwise runs forever.')
    args = parser.parse_args()
    with PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as output:
            
        
            # Forced sensor mode, 1640x1232, full FoV. See:
            # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
            # This is the resolution inference run on.
            camera.sensor_mode = 4

            # Scaled and cropped resolution. If different from sensor mode implied
            # resolution, inference results must be adjusted accordingly. This is
            # true in particular when camera.start_recording is used to record an
            # encoded h264 video stream as the Pi encoder can't encode all native
            # sensor resolutions, or a standard one like 1080p may be desired.
            camera.resolution = (1640, 922)

            # Start the camera stream.
            camera.framerate = 30
           
            
            camera.start_preview()
            
        
            # Annotator renders in software so use a smaller size and scale results
            # for increased performace.
    ##        annotator = Annotator(camera, dimensions=(320, 240))
            scale_x = 320 / 1640
            scale_y = 240 / 1232

            # Incoming boxes are of the form (x, y, width, height). Scale and
            # transform to the form (x1, y1, x2, y2).
            def transform(bounding_box):
                x, y, width, height = bounding_box
                return (scale_x * x, scale_y * y, scale_x * (x + width),
                        scale_y * (y + height))
            flag =0
            #stream = io.BytesIO()
            with CameraInference(face_detection.model()) as inference:
                #print(inference._engine)
                
                    
                for i, result in enumerate(inference.run()):
                    if i == args.num_frames:
                        break
                    
                    
                    
                    faces = face_detection.get_faces(result)
    ##                if len(faces)>=1:
    ##                    camera.capture('NewImg'+str(i)+'.jpg')

    ##                annotator.clear()
                    for face in faces:
                        bbox = face.bounding_box
                        print(bbox)
    ##                    annotator.bounding_box(bbox, fill=0)
                        camera.capture(output,'rgb')
                        print('output array:',output.array.shape)
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[0]+bbox[2])
                        y2 = int(bbox[1]+bbox[3])
                        face_sq = output.array[y1:y2,x1:x2]
                        print(face_sq.shape)
                        im = Image.fromarray(face_sq)
                        im.save("you_file"+str(i)+".jpg")
                        output.truncate(0)
    ##                    frame_buffer = annotator._buffer
    ##                    print('buffer type:',type(frame_buffer))
    ##                    print('buffer dims:', annotator._buffer_dims)
    ##                    print('bbox data:', bbox)
    ##                    frame_face = frame_buffer.crop(box=bbox)
    ##                    frame_data = np.asarray(frame_face)
    ##                    print('face dims:',frame_data.shape)
    ##                    print('face data:', frame_data)
    ##                    if flag == 1:
    ##                        frame_buffer.save("NewImg.png")
    ##                        flag == 0
    ##                annotator.update()
                    print('Iteration #%d: num_faces=%d' % (i, len(faces)))
           
                    
            camera.stop_preview()


if __name__ == '__main__':
    main()

