import cv2
import time
if __name__=='__main__':
    video = cv2.VideoCapture(0)
    video.set(3,640)
    video.set(4,480)
    fps=video.get(cv2.CAP_PROP_FPS)
    print("frames from function:{0}".format(fps))
    num_frames = 120
    print("capture {0} frames".format(num_frames))
    start = time.time()
    for i in range(0, num_frames):
        ret, frame = video.read()
        #print(frame.shape)
    end = time.time()
    seconds = end- start
    print("time taken: {0}".format(seconds))
    fps = num_frames/seconds
    print("frames from testing:{0}".format(fps))
    video.release()