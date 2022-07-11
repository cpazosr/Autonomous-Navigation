#!/usr/bin/env python3
# Original File          :   Pokenet.py (Pokenet's testing script)
# Original Author        :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# Version                :   1.0.2
# Description            :   Script that calls the Pokenet CNN and tests it on example images.
# License                :   MIT
###########################################################################################
# Modiffied File         :   classify2.py (sign detector topic publisher script)
# Date:                  :   June 17, 2022
# Adaptation Authors     :   Manuel Agustin Diaz & Carlos Antonio Pazos
# Description            :   Script that calls the CNN to classifyimages an publish the results in a ros topic.


# Import the necessary packages:
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import rospy
from tensorflow.keras import backend as bk
from tensorflow.config import set_visible_devices
from std_msgs.msg import Float32,String
from sensor_msgs.msg import Image
import os

class Classify():
	def __init__(self):

		bk.clear_session()
		set_visible_devices([],'GPU')

		# set the resources paths:
		mainPath = os.path.join("/home/puzzlebot")
		modelPath = os.path.join(mainPath, "output2")
		# open CNN model
		self.model = load_model(os.path.join(modelPath, "signs2.model"))
		# initialize nodes
		rospy.init_node("classify")
		# initialize publishers
		self.class_pub = rospy.Publisher("/classify/class",String,queue_size=1)
		self.prob_pub = rospy.Publisher("/classify/prob",Float32,queue_size=1)
		# self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber("/processed_image/crop",Image,self.image_Callback)
		# rate
		self.rate = rospy.Rate(30)
		# initialize rate as an empty uin8 array
		self.frame = np.array([[]], dtype = "uint8")
		# initialize previous class save
		self.prev = "0.0"

	def imgmsg_to_cv2(self,msg):
		# transform imgmsg to a variable accepted by cv2
		return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)


	def image_Callback(self,data):
		# retrieve camera image data
		try:
			frame = self.imgmsg_to_cv2(data)
			self.frame = frame
		except Exception as e:
			# if an error happens, print the error
			print(e)



	def pubClass(self,prob,label):
		# function tu publish label and probability
		# to the topics /classify/class and /classify/prob respectively
		self.class_pub.publish(label)
		self.prob_pub.publish(prob)

	def image_classify(self):
		# function that opens the model and predicts the class
	    # save image in a variable
		image = self.frame
		# image shape
		height = image.shape[0]
		width = image.shape[1]

		#new height and width
		# Training image size:
		imageSize = (100, 100)
		# The class dictionary:
		# derecho               1000
		# fin_reglas            0100
		# stop                  0010
		# vuelta_derecha        0001
		classDictionary = { 0: "derecho", 1: "fin_reglas", 2: "stop", 3: "vuelta_derecha", }

		# pre-process the image for classification
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, imageSize)
		image = image.astype("float") / 255.0

		# add the "batch" dimension:
		image = np.expand_dims(image, axis=0)

		# send to CNN, get probabilities:
		predictions = self.model.predict(image)
		# get max probability, thus, the max classification result:
		classIndex = predictions.argmax(axis=1)[0]

		# filter everything with 80 or less probability
		if float(predictions[0][classIndex]) > 0.80:
			label = classDictionary[classIndex] # Get categorical label via the class dictionary
		else:
			# if the probability is less than 80, the labes is changed to No signal
			label = "No signal"

		# print the classification result:
		print("Class: " + label + " prob: " + str(predictions[0][classIndex]))

		# build the label and draw the label on the image
		prob = "{:.2f}".format(predictions[0][classIndex])
		label = label

		# publish the classification only if it is different from the previous one
		if self.prev != prob:
			# also, if the labes is different from No signal, publish the label
			# in the topics /classify/class and /classify/prob
			if label != "No signal":
				self.pubClass(float(prob), label)
		self.prev = prob

	def main(self):
		# run function image_classify as long as the node is running
		while not rospy.is_shutdown():
			try:
				self.image_classify()
			except Exception as e:
				# print error if precented
				# Note: depending of the order in the launch sequence
				# the code might detect a cv2 empty error
				# just run your other nodes and this error should disapear
				print(e)
				print("wait")
			self.rate.sleep()



if __name__ == '__main__':
	try:
		# create Classify object
		img = Classify()
		# run main of Classify
		img.main()
	except (rospy.ROSInterruptException, rospy.ROSException('Topic was interrupted')):
		pass
