#!/usr/bin/env python
import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv_bridge
import numpy as np

class LightsDetector():
	def __init__(self):
		# node constructor with publishers, subscribers and class variables
		rospy.init_node("lights_detector")
		#self.img_pub = rospy.Publisher("processed_image/light",Image,queue_size=1)
		self.rth_pub = rospy.Publisher("/red_thresh",Image,queue_size=1)
		self.gth_pub = rospy.Publisher("/green_thresh",Image,queue_size=1)
		self.str_pub = rospy.Publisher("/light",String,queue_size=1)
		self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber("/video_source/raw",Image,self.image_Callback)
		self.rate = rospy.Rate(30)
		self.frame = np.array([[]], dtype = "uint8")
		self.light = String()
		self.light = ""

	def image_Callback(self,data):
		# image from subscription
		try:
			frame = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
			self.frame = frame
		except cv_bridge.CvBridgeError():
			print("Error bridge cv")

	def image_process(self):
		
		# copy from image subscriber
		frame = self.frame
		# -- correct barrel distortion from camera --
		width  = frame.shape[1]
		height = frame.shape[0]
		# vector of coeffs
		distCoeff = np.zeros((4,1),np.float64)
		# callibration
		k1 = -1.0e-3;	# negative to remove barrel distortion
		k2 = -1.0e-7;
		p1 = 1.0e-3;
		p2 = 1.0-1;
		distCoeff[0,0] = k1;
		distCoeff[1,0] = k2;
		distCoeff[2,0] = p1;
		distCoeff[3,0] = p2;
		# assume unit matrix for camera
		cam = np.eye(3,dtype=np.float32)
		cam[0,2] = width/2.0  # define center x
		cam[1,2] = height/2.0 # define center y
		cam[0,0] = 10.        # define focal length x
		cam[1,1] = 10.        # define focal length y
		# undistortion
		dst = cv2.undistort(frame,cam,distCoeff) # frame, camMatrix, vector of undistortion coefs
		
		#img = self.bridge.cv2_to_imgmsg(dst,"bgr8")

		# crop a bit of right side
		new_width = int(width * 0.85)
		dst = dst[0:height, 0:new_width]	# [row:row,col:col]
		
		# change to HSV colors
		hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)

		# HSV ranges
		lower_green = np.array([49,35,122])
		higher_green = np.array([93,255,255])
		#lower_yellow = np.array([21,34,104])
		#higher_yellow = np.array([57,255,255])
		lower_red = np.array([0,10,210])
		higher_red = np.array([55,210,255])

		# HSV masks in range
		redMask = cv2.inRange(hsv, lower_red, higher_red)
		#yellowMask = cv2.inRange(hsv, lower_yellow, higher_yellow)
		greenMask = cv2.inRange(hsv, lower_green, higher_green)

		# thresholding with otsu
		red_n, redThresh = cv2.threshold(redMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#yellow_n, yellowThresh = cv2.threshold(yellowMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		green_n, greenThresh = cv2.threshold(greenMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# morphological operations by mask
		#kernel = np.ones((4,4),np.uint8)
		redThresh = cv2.erode(redMask,None,iterations = 0)
		redThresh = cv2.dilate(redThresh,None,iterations = 4)
		redThresh = cv2.GaussianBlur(redThresh,(5,5),0)
		#yellowThresh = cv2.erode(yellowThresh,kernel)
		#yellowThresh = cv2.dilate(yellowThresh,kernel)
		greenThresh = cv2.erode(greenMask,None,iterations = 1)
		greenThresh = cv2.dilate(greenThresh,None,iterations = 10)
		greenThresh = cv2.GaussianBlur(greenThresh,(5,5),0)
		
		# publish processed images
		rth = self.bridge.cv2_to_imgmsg(redThresh,"passthrough")
		gth = self.bridge.cv2_to_imgmsg(greenThresh,"passthrough")
		self.rth_pub.publish(rth)
		self.gth_pub.publish(gth)
		
		# parameters for filtering with blob detection
		params = cv2.SimpleBlobDetector_Params()
		# by area
		params.filterByArea = True
		params.minArea = 100
		# by color & threshold
		params.filterByColor = True
		params.blobColor = 255
		params.minThreshold = 0
		params.maxThreshold = 100
		# by circularity
		params.filterByCircularity = True
		params.minCircularity = 0.8

		# blob object
		blobDetector = cv2.SimpleBlobDetector_create(params)
		# detect blobs as lists per mask with same blob detector
		redBlobs = blobDetector.detect(redThresh)
		#yellowBlobs = blobDetector.detect(yellowThresh)
		greenBlobs = blobDetector.detect(greenThresh)

		# check for 1 blob if present in any mask
		img = None
		if len(greenBlobs) == 1:
			# green light
			print("Green Light, Go")
			try:
				# publish only when light changes
				if self.light != "green":
					img = self.bridge.cv2_to_imgmsg(greenThresh,"passthrough")
					self.light = "green"
			except Exception as e:
				print("greenThresh pub exeption")
				print(e)
		#elif len(yellowBlobs) == 1:
			#print("Yellow Light, Slow")
			#try:
				#if self.light != "yellow":
					#img = self.bridge.cv2_to_imgmsg(yellowThresh,"passthrough")
					#self.light = "yellow"
			#except Exception as e:
				#print("yellowThresh pub exception")
				#print(e)
		elif len(redBlobs) == 1:
			# red light
			print("Red Light, Stop")
			try:
				# publish only when light changes
				if self.light != "red":
					img = self.bridge.cv2_to_imgmsg(redThresh,"passthrough")
					self.light = "red"
			except:
				print("redThresh pub excpetion")
				print(e)
		else:
			# no traffic light
			pass
		
		# publish light only if light detected and changed
		if img != None:
			#self.img_pub.publish(img)
			self.str_pub.publish(self.light)

	def main(self):
		# run while node active
		while not rospy.is_shutdown():
			try:
				self.image_process()
			except Exception as e:
				print(e)
				print("wait")
			self.rate.sleep()


if __name__ == '__main__':
	try:
		img = LightsDetector()
		img.main()
	except (rospy.ROSInterruptException, rospy.ROSException('Topic was interrupted')):
		pass
