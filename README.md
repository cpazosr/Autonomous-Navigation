# Autonomous-Navigation
ROS package for Puzzlebot Autonomous Navigation

This repository contains the code for the Puzzlebot robot to navigate a track fully autonomous, based on traffic lights and traffic signs along the trajectory. This solution was developed for the Puzzlebot, a great robotic invention provided by @Manchester-Robotics for learning and exploring robotics. The code runs in the NVIDIA Jetson Nano mounted on the robot, specialized hardware enhanced for computer vision, robotics and many other GPU operations.

Learn more about Puzzlebot and Manchester Robotics here: https://www.manchester-robotics.com/

This repository serves as the source code solution to the challenge designed by Manchester Robotics and teachers from Tecnologico de Monterrey for subject Implementation of Intelligent Robotics. The challenge consisted in a shaped "b" track with traffic lights and traffic signs as shown below:

![imagen](https://user-images.githubusercontent.com/67598380/178133376-d08daeb8-d021-41a1-9a15-109982ec8528.png) 
![imagen](https://user-images.githubusercontent.com/67598380/178133382-d9398da2-9137-4aa6-a825-b826068e811d.png)
> Manchester Robotics (2022)

Puzzlebot must always keep on track, following the line and crossing the intersection depending on the traffic light (red or green) and on the traffic sign read before crossing. There are four signs to read, but the code is open to add more.
![imagen](https://user-images.githubusercontent.com/67598380/178134067-7fa268fd-848a-4043-955c-6462d7069500.png)

This solution focuses efforts on robotics and computer vision with control engineering and fundamentals of artificial intelligence. It combines simultaneous nodes in ROS using Melodic distribution to detect multiple things in the track and control the robot reactive actions based on that. All detection was made with a Pi Camera connected to Jetson Nano with a modified resolution of 240x120 for an approximate real time operation.

## Solution 

You can check out a video of the solution here: https://drive.google.com/file/d/1Qn0xJXrjO7snd7zQHDGnzeJXvBT6Cl9z/view?usp=sharing

Solution by @AgusDV111 and me with Manchester Robotics Puzzlebot and tested in Tecnologico de Monterrey campus CEM.

This repository contains all files to run the ROS nodes to run the solution. The next files run each one a ROS node:
- classify2.py
- controller.py
- lights_detector.py
- line_detector.py
- sign_detector.py

For you to run these files on your hardware you must download or copy each of the files above to the *src* directory of your ROS package. For example our workspace was named *catkin_ws* and the ROS package was named *simple_vision*, so the files were located in */catkin_ws/src/simple_vision/src*

The other Python file *model_generator.py* is the code for creating the CNN (Convolutional Neural Network) model to classify the detected signs. The model used in the presented solution is included under the directory name *output2* as you need to install and configure your computer to run the model generator, specifically tensorflow, but it depends on your machine. Do consider that you should install and check tensorflow requirements and other dependencies in your hardware before running your model, as *classify2.py* uses these libraries as well. This solution uses Python3 for CNN model related processing, but Python2 for other ROS actions. Also consider the location of your model as you'll need to reference the path in the classifier node. We located it in the */home* directory of the robot.

The other file is the rqt plot presented with ROS, that shows the relation taken place between the nodes and finally the robot. Initially, the robot has a topic to broadcast video captured by the camera as */video_source/raw* which is received by the nodes */lights_detector*, */line_detector*, and */sign_detector* which process everything of interest and eventually send it to the */controller* node for the robot to move and take action respectively by publishing in the */cmd_vel* topic.

Each part is explained further below.

## Sign Detection

## CNN model and Classification

## Line Follower and Intersections

## Traffic Lights Detection

## Controller

## Final Comments
