import argparse
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from picamera import PiCamera
import picamera
import picamera.array
import numpy as np
from PIL import Image
import os
import numpy
import numpy as np
import lingyu_pi_head


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
            camera.resolution = (1640, 1232)

            # Start the camera stream.
            camera.framerate = 30
            camera.start_preview()

            with CameraInference(face_detection.model()) as inference:
                # Lingyu, get raw data
                # Important note: camera.capture() can not be used together with Annotator
                for i, result in enumerate(inference.run()):
                    if i == args.num_frames:
                        break
                    faces = face_detection.get_faces(result)
                    for face in faces:
                        bbox = face.bounding_box
                        #print(bbox)
                        camera.capture(output,'rgb')
                        #print('output array:', output.array.shape)
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[0]+bbox[2])
                        y2 = int(bbox[1]+bbox[3])
                        face_img = output.array[y1:y2,x1:x2]
                        # im = Image.fromarray(face_sq)
                        # im.save("you_file"+str(i)+".jpg")
                        output.truncate(0)
                        # Lingyu. Plug in the face bounding box as the input of head pose
                        # Lingyu. Get square isolated face data from bbox data
                        cam_w = int(bbox[2])
                        cam_h = int(bbox[3])
                        fine_size = min(cam_w,cam_h)
                        face_sq_np = face_img[(cam_h//2 - fine_size//2): (cam_h//2 + fine_size//2), (cam_w//2 - fine_size//2): (cam_w//2 + fine_size//2)]
                        size = 64, 64
                        face_sq_raw = (Image.fromarray(face_sq_np,'RGB')).resize(size)
                        #face_sq_raw.save('out.jpg')
                        #face_sq = np.array(face_sq_raw)
                        #print(face_sq.shape)
                        face_sq = face_sq_raw
##                        roll = lingyu_pi_head.return_roll(face_sq) # Evaluate the roll angle using a CNN
##                        pitch = lingyu_pi_head.return_pitch(face_sq) # Evaluate the pitch angle using a CNN
##                        yaw = lingyu_pi_head.return_yaw(face_sq)  # Evaluate the yaw angle using a CNN
##                        print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
##                        print("")
                        # Lingyu. Plug in the face bounding box as the input of head pose

                       
                        
                        
                    print('Iteration #%d: num_faces=%d' % (i, len(faces)))
                    
                    
                    
                    
        camera.stop_preview()


if __name__ == '__main__':
    main()
