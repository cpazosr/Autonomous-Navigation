#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import Float32,Bool,String

# callbacks
def end_Callback():
	# stop at last
	robot_cmd.linear.x = 0.0
	robot_cmd.angular.z = 0.0
	cmd_pub.publish(robot_cmd)

def camCallback(msg):
	# camera center
	global cam_x
	global cam_y
	cam_x = msg.x
	cam_y = msg.y

def boxCallback(msg):
	# box center
	global box_x
	global box_y
	global box_theta
	box_x = msg.x
	box_y = msg.y
	box_theta = msg.theta

def zebraCallback(msg):
	# crosswalk
	global zebra
	zebra = msg.data

def onesCallback(msg):
	# lines
	global line
	line = msg.data

def zeroCallback(msg):
	# zero contours
	global zero
	zero = msg.data

def classCallback(msg):
	# sign class
	global label
	label = msg.data

def probCallback(msg):
	# sign class probability
	global prob
	prob = msg.data

def lightCallback(msg):
	# light color
	global light
	light = msg.data


# init node
rospy.init_node("controller")
# publisher and subscribers and variables
cmd_pub = rospy.Publisher("/cmd_vel",Twist,queue_size = 1)
rospy.Subscriber("/cam_center",Pose2D,camCallback)
rospy.Subscriber("/box_center",Pose2D,boxCallback)
rospy.Subscriber("/zebra",Bool,zebraCallback)
rospy.Subscriber("/line",Bool,onesCallback)
rospy.Subscriber("/zero",Bool,zeroCallback)
rospy.Subscriber("/classify/class",String,classCallback)
rospy.Subscriber("/light",String,lightCallback)
rospy.Subscriber("/classify/prob",Float32,probCallback)
rate = rospy.Rate(60)
robot_cmd = Twist()
cam_x = float()
cam_y = float()
box_x = float()
box_y = float()
box_theta = float()
light = ""
zebra = False
line = False
zero = False
label = ""
prob = float()
in_crosswalk = False
continue_line = False
wait_green = False
in_process = ""

# functions for robot movement
def stop():
	# stop robot
	print("stoping...")
	robot_cmd.linear.x = 0.0
	robot_cmd.angular.z = 0.0
	cmd_pub.publish(robot_cmd)

def cross_straight(maxvel=0.05):
	# passing crosswalk straight
	print("crosswalk straight...")
	robot_cmd.angular.z = 0
	robot_cmd.linear.x = maxvel
	cmd_pub.publish(robot_cmd)

def cross_right(maxvel=0.07,R=2):
	# passing crosswalk turning right
	print("crosswalk right...")
	robot_cmd.linear.x = maxvel
	robot_cmd.angular.z = -0.045	# (1/R)*maxvel -> ideal circular turn
	cmd_pub.publish(robot_cmd)

def follow(maxvel=0.06, K=0.0009):
	# following line based on box and camera centers
	print("following line...")
	# proportional gain for control of the difference centers coordinates
	x_dif = (box_x - cam_x) * -K
	y_dif = (box_y - cam_y) * -K
	# incoming turn -> slow linear velocity
	if (box_theta < -75 and box_theta > -80) or (box_theta < -15 and box_theta > -20):
		robot_cmd.linear.x = maxvel/2
	else:
		robot_cmd.linear.x = maxvel
	# angular speed saturation
	if x_dif < -0.2:
		robot_cmd.angular.z = -0.2
	elif x_dif > 0.2:
		robot_cmd.angular.z = 0.2
	# move towards the center of line detected -> adjust angular vel
	else:
		robot_cmd.angular.z = x_dif
	cmd_pub.publish(robot_cmd)

def main():
	# states variables
	global in_crosswalk
	global continue_line
	global wait_green
	global in_process
	global light
	
	# check if many 1-contours are detected -> line detected
	if line:
		continue_line = True	# follow line
		in_crosswalk = False	# not crossing intersection
	
	# if crosswalk detected and not already inside intersection
	if zebra and not in_crosswalk:
		print("entered crosswalk...")
		in_crosswalk = True		# entered intersection
		if light == "red":		# wait at crosswalk if no green light
			wait_green = True
		else:
			wait_green = False	# already green light
		in_process = label		# do not change sign while crossing intersection
	
	# if crosswalk detected and already inside intersection
	elif zebra and in_crosswalk:
		print("exited crosswalk...")
		in_crosswalk = False	# exited intersection
		wait_green = False		# no more traffic lights
		light = "green"			# always keep movement

	# if stopped at crosswalk due to red light
	if wait_green:
		print("wating for green")
		if light == "green":		# traffic light updates to green
			wait_green = False		# wait no more
		else:
			print("stop, wating for green light")
			stop()		# stop while red light

	# handling actions
	if in_crosswalk and wait_green == False:	# crossing intersection
		continue_line = False	# never follow line 
		print(light + " :stop_go")
		if in_process == "vuelta_derecha":	# detected sign is turn right
			cross_right()		# turn right
		elif in_process == "derecho":		# detected sign is go forward
			cross_straight()	# move forward
		else:
			stop()		# no forward or turn right sign detected -> no decision so stop
	# if STOP sign detected, stop until you see no line to follow
	elif label == "stop" and zero == True:
		print("Entramos al stop por senyal")
		stop()
	# detected sign: no rules
	elif label == "fin_reglas":
		print("fin de reglas")
		follow(0.11,0.0018)		# follow line but a bit faster
	# line detected and no red light
	elif continue_line and wait_green == False:
		follow()	# should follow line
	else:
		pass	# no action


#------------------------Main-------------------------------
print("Running main ...")
while not rospy.is_shutdown():
	# while node active
	try:
		main()
	except rospy.ROSInterruptException:
		rospy.on_shutdown(end_Callback)
		print("ROS Interrupt Exception, done by user")
	except rospy.ROSTimeMovedBackwardsException:
		rospyu.logerr("ROS Time Backwards! just ignore")
	rate.sleep()
end_Callback()
