#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool
import cv_bridge

class LineFollowerNode():

	def __init__(self):
		# constructor node publishers and subscribers and class variables
		rospy.init_node("line_detector")
		self.img_pub = rospy.Publisher("/processed_image/line",Image,queue_size=1)
		self.zebra_pub = rospy.Publisher("/zebra",Bool,queue_size=1)
		self.ones_pub = rospy.Publisher("/line",Bool,queue_size=1)
		self.zero_pub = rospy.Publisher("/zero",Bool,queue_size=1)
		self.bridge = cv_bridge.CvBridge()
		self.img_sub = rospy.Subscriber("/video_source/raw",Image,self.imgCallback)
		self.cam_origin_pub = rospy.Publisher("/cam_center",Pose2D,queue_size=1)
		self.box_origin_pub = rospy.Publisher("/box_center",Pose2D,queue_size=1)
		self.rate = rospy.Rate(60)
		self.frame = np.array([[]],dtype="uint8")
		self.cam_origin = Pose2D()
		self.box_origin = Pose2D()
		self.after_zero = []
		self.only_ones = []

	def imgCallback(self,data):
		# callback for the img from camera
		try:
			frame = self.bridge.imgmsg_to_cv2(data,desired_encoding="bgr8")
			self.frame = frame
		except cv_bridge.CvBridgeError():
			print("Error CvBridge")

	def processImg(self):
		# copy of image from subscriber
		frame = self.frame
		dst = frame

		# crop frame
		height = dst.shape[0]	# rows
		width = dst.shape[1]	# cols
		new_height = int(height * 0.85)	# 85%
		new_width = int(width * 0.65)	# 65%
		dst = dst[new_height:height, 0+(width-new_width)/2:width-(width-new_width)/2]	# [row:row,col:col]

		# grayscale to binary (white line) with otsu thresholding
		gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

		# get center of cropped frame
		origin_x = int(new_width/2)
		origin_y = int((height-new_height)/2)	#int(new_height/2)
		cv2.circle(dst,(origin_x,origin_y),2,(0,255,0),1)
		self.cam_origin.x = origin_x
		self.cam_origin.y = origin_y
		self.cam_origin_pub.publish(self.cam_origin)

		# blur to binary (white line) thresholding
		blur = cv2.GaussianBlur(gray,(5,5),0)
		val,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY_INV)

		# filter out noise
		# find all blobs (white binary)
		threshold = 150		# takeout size
		nb_components,output,stats,centroids = cv2.connectedComponentsWithStats(thresh,connectivity=8)	# numComponents, components, stats, centroids // type 4 or 8
		sizes = stats[1:, -1]; nb_components = nb_components - 1	# take sizes
		img = np.zeros((output.shape),dtype = np.uint8)	# image carrier
		# filter by threshold size
		for i in range(0,nb_components):
			if sizes[i] >= threshold:
				img[output == i + 1] = 255	# black the bounding rect of component

		# contours examination
		contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		cv2.drawContours(dst,contours,-1,(255,0,0),2)
		count = len(contours)
		print( str(count)+' contours' )
		# minimum of 1 contours
		if count >= 1:
			# robot seeing constant line
			if count == 1:
				self.only_ones.append(1)
			else:
				self.only_ones = []
			# append for entering crosswalk
			if len(self.after_zero) >= 1 and self.after_zero[0] == 0:
				self.after_zero.append(count)
			# check out 3 or 4 contours to confirm crosswalk detection
			if 3 in self.after_zero or 4 in self.after_zero:
				self.zebra_pub.publish(True)
				print("zebra detected")
				self.after_zero = []
			else:
				self.zebra_pub.publish(False)
			# crosswalk saturation -> restart accumulator
			if len(self.after_zero) >= 65:
				self.after_zero = []
			# line detection after 85 consecutive 1s detected
			if len(self.only_ones) >= 85:
				print("follow line")
				self.ones_pub.publish(True)
				self.only_ones = []
			else:
				self.ones_pub.publish(False)
			# max contour -> black line detected
			c = max(contours,key=cv2.contourArea)
			# min enclosing area rectangle
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			# draw on frame
			cv2.drawContours(dst,[box],0,(0,0,255),2)
			# get rectangle centroid
			box_x = int(rect[0][0])
			box_y = int(rect[0][1])
			box_theta = rect[2]
			# draw on frame
			cv2.circle(dst,(box_x,box_y),2,(0,0,255),2)
			# publish
			self.box_origin.x = box_x
			self.box_origin.y = box_y
			self.box_origin.theta = box_theta
			self.box_origin_pub.publish(self.box_origin)
			self.zero_pub.publish(False)
		# 0 contours detected
		else:
			self.after_zero = []
			self.after_zero.append(0)
			self.only_ones = []
			self.zero_pub.publish(True)
		# monitoring
		print('after_zeros:',self.after_zero)
		print('only ones:',self.only_ones)
		self.img_pub.publish(self.bridge.cv2_to_imgmsg(dst,'bgr8'))

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
		node = LineFollowerNode()
		node.main()
	except (rospy.ROSInterruptException, rospy.ROSException("Topic interrupted")):
		pass
