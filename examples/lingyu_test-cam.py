import time
import io
import picamera
import picamera.array

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution= (100,100)
        camera.start_preview()
        time.sleep(2)
        camera.capture(stream, 'rgb')
        print(stream.array.shape)