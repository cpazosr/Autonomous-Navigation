#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv_bridge

class SignDetectorNode():

	def __init__(self):
		# constructor node publishers and subscribers
		rospy.init_node("sign_detector")
		self.img_pub_output = rospy.Publisher("/processed_image/output",Image,queue_size=1)
		self.img_pub_crop = rospy.Publisher("/processed_image/crop",Image,queue_size=1)
		self.bridge = cv_bridge.CvBridge()
		self.img_sub = rospy.Subscriber("/video_source/raw",Image,self.imgCallback)
		self.rate = rospy.Rate(60)
		self.frame = np.array([[]],dtype="uint8")

	def imgCallback(self,data):
		# callback for the img from camera
		try:
			frame = self.bridge.imgmsg_to_cv2(data,desired_encoding="bgr8")
			self.frame = frame
		except cv_bridge.CvBridgeError():
			print("Error CvBridge")

	def filter_colors(self,img):
		# blur and HSV
		frame = cv2.GaussianBlur(img,(3,3),0)
		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		# -- ranges --
		# blue
		lower_blue = np.array([100,128,0],dtype=np.uint8)
		upper_blue = np.array([215,255,255],dtype=np.uint8)
		# -- masks --
		mask_blue = cv2.inRange(hsv,lower_blue,upper_blue)

		return mask_blue

	def processImg(self):

		# copy of image subscriber
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

		# copy for writing final output on it
		outputCrop = dst.copy()

		# -- preprocess --
		# increase contrast and dynamic range
		lab = cv2.cvtColor(dst,cv2.COLOR_BGR2LAB)
		l,a,b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		cl = clahe.apply(l)
		merged = cv2.merge((cl,a,b))
		saturated = cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)
		# blurred edges with Gaussian and Laplacian
		blur = cv2.GaussianBlur(saturated, (3,3), 0)	# blur
		gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)	# gray
		laplacian = cv2.Laplacian(gray,cv2.CV_8U,5,3,1,cv2.BORDER_ISOLATED)	# edges #original CV_8U
		laplacian = cv2.convertScaleAbs(laplacian)	# abs values
		# binarize
		_,thresh = cv2.threshold(laplacian,30,255,cv2.THRESH_BINARY)

		# -- filter by color --
		colors = cv2.bitwise_and(thresh,thresh,mask=self.filter_colors(frame))

		# -- filter out small blobs --
		# find all blobs (white binary)
		threshold = 150		# takeout size
		nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(thresh,connectivity=8)	# numComponents,components, stats, centroids // type 4 or 8
		sizes = stats[1:, -1]; nb_components = nb_components - 1	# take sizes
		img = np.zeros((output.shape),dtype = np.uint8)	# image carrier
		# filter by threshold size
		for i in range(0,nb_components):
			if sizes[i] >= threshold:
				img[output == i + 1] = 255	# black the bounding rect of component
		
		# -- find sign from contours --
		# contours
		contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		# evaluate contours
		for c in contours:
			perimeter = cv2.arcLength(c,True)	# perimeter
			approx = cv2.approxPolyDP(c,0.02*perimeter,True)	# polygon
			# check for sides and convexity
			if len(approx) == 8:
				conv = cv2.isContourConvex(approx)	# circle -> convex
				if conv:	# detection
					print("sign")
					x,y,w,h = cv2.boundingRect(c)	# detection layout
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)	# draw detection
					# crop
					x += int(w/2)	# center x
					y += int(h/2)	# center y
					factor = 1.25	# crop factor widening
					new_w = int(np.round( (w*factor)/2,0 ))	# wider width
					new_h = int(np.round( (h*factor)/2,0 ))	# wider height
					outputCrop = outputCrop[y-new_h:y+new_h,x-new_w:x+new_w]	# crop
					try:
						self.img_pub_crop.publish(self.bridge.cv2_to_imgmsg(outputCrop,"bgr8"))
					except Exception as e:
						print("Empty outputCrop")
						print(e)
		self.img_pub_output.publish(self.bridge.cv2_to_imgmsg(frame,"bgr8"))


	def main(self):
		# main execution while node runs
		print("Running main...")
		while not rospy.is_shutdown():
			try:
				self.processImg()
			except Exception as e:
				print("wait")
				print(e)
			self.rate.sleep()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	try:
		node = SignDetectorNode()
		node.main()
	except (rospy.ROSInterruptException, rospy.ROSException("Topic interrupted")):
		pass
