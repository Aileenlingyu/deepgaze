#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import tensorflow as tf
import cv2
import numpy
import numpy as np
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_roll_variables(os.path.realpath("../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))

# for i in range(1,9):
#     file_name = str(i) + ".jpg"
#     print("Processing image ..... " + file_name)
#     image = cv2.imread(file_name) #Read the image with OpenCV
#     print(image.shape)
#     # Get the angles for roll, pitch and yaw
#     roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
#     pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
#     yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
#     print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
#     print("")
def angle2rotmat(x,y,z):
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
def main():

	#Defining the video capture object
	video_capture = cv2.VideoCapture(0)
	if(video_capture.isOpened() == False):
	    print("Error: the resource is busy or unvailable")
	else:
	    print("The video source has been opened correctly...")

	#Create the main window and move it
	cv2.namedWindow('Video')
	cv2.moveWindow('Video', 20, 20)
	
	#Obtaining the CAM dimension
	cam_w = int(video_capture.get(3))
	cam_h = int(video_capture.get(4))

	c_x = cam_w / 2
	c_y = cam_h / 2
	f_x = c_x / np.tan(60/2 * np.pi / 180)
	f_y = f_x

	#Estimated camera matrix values.
	camera_matrix = numpy.float32([[f_x, 0.0, c_x],
	                               [0.0, f_y, c_y], 
	                               [0.0, 0.0, 1.0] ])

	print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")

	# #These are the camera matrix values estimated on my webcam with
	# # the calibration code (see: src/calibration):
	# camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
	#                                [         0.0, 603.55869786,  229.7537026], 
	#                                [         0.0,          0.0,          1.0] ])

	#Distortion coefficients
	#camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

	#Distortion coefficients estimated by calibration
	camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])
	axis = numpy.float32([[50,0,0], 
	                                  [0,50,0], 
	                                  [0,0,50]])

	tvec = np.array([1.0, 1.0, 1.0], np.float) # translation vector

	while(True):

		# Capture frame-by-frame
		
		ret, frame = video_capture.read()
		#image = cv2.resize(frame,(480,480),interpolation = cv2.INTER_CUBIC)
		image = frame[(cam_h//2 - 480//2): (cam_h//2 + 480//2), (cam_w//2 - 480//2): (cam_w//2 + 480//2)]
		roll = my_head_pose_estimator.return_roll(image) # Evaluate the roll angle using a CNN
		pitch = my_head_pose_estimator.return_pitch(image) # Evaluate the pitch angle using a CNN
		yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
		# roll = my_head_pose_estimator.return_roll(image, radians=True) # Evaluate the roll angle using a CNN
		# pitch = my_head_pose_estimator.return_pitch(image, radians=True) # Evaluate the pitch angle using a CNN
		# yaw = my_head_pose_estimator.return_yaw(image, radians=True)  # Evaluate the yaw angle using a CNN
		print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
		print("")

		rot_matrix = angle2rotmat(-roll[0,0,0],-pitch[0,0,0],-yaw[0,0,0])
		rvec, jacobian = cv2.Rodrigues(rot_matrix)
		imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
		p_start = (int(c_x), int(c_y))
		p_stop = (int(imgpts[2][0][0]), int(imgpts[2][0][1]))
		
		# # cv2.line(image, p_start, p_stop, (0,0,255), 3) #Red
		# cv2.circle(image, p_start, 1, (0,255,0), 3) #Green

		# cv2.line(image, p_start, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0,255,0), 3) #GREEN
		# cv2.line(image, p_start, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (255,0,0), 3) #BLUE
		# cv2.line(image, p_start, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (0,0,255), 3) #RED

		cv2.putText(image, "X: " +  "{:7.2f}".format(roll[0,0,0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
		cv2.putText(image, "Y: " + "{:7.2f}".format(pitch[0,0,0]), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
		cv2.putText(image, "Z: " + "{:7.2f}".format(yaw[0,0,0]), (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

		cv2.imshow('Video', image)
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	video_capture.release()
	print("Bye...")
if __name__ == "__main__":
    main()
