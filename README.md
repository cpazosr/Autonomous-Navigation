# Autonomous-Navigation
ROS package for Puzzlebot Autonomous Navigation

Authors: cpazosr & AgusDV111

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

This is a node dedicated to receiving the raw video capture to locate and crop the desired traffic signs to send to classify and eventually take movement decisions. This method is based on borders detection and contours, with strong filtering by sizes, color and thresholds.
Processing is as follows:
1. Camera correction: using cv2.undistort with manual matrix configuration to correct barrel distortion from camera lens.
2. Contrast augmentation: increase contrast with CLAHE method to the brightness channel in LAB colorspace.
3. GaussianBlur and Grayscale.
4. Laplacian edges: tryout different parameter configurations.
5. Binarization: fixed thresholding.
6. Color filter: bitwise AND to filter out all borders not included in the HSV specified range.
7. Size filter: using cv2.connectedComponentsWithStats to get each blob size and take out all blobs under a certain size threshold.
8. Find contours.
9. Approximate polygon: find closed polygon per contour.
10. Check sides: from the polygon found, check if it is considered to have 8 sides, which is the number of sides considered for a circle. With 8 sides, also STOP signal (hexagon) is considered.
11. Check convexity: to return only circles and closed hexagonal figures as these have convexity.
12. Get bounding rectangle: after all filtering, if there is a contour with the previous conditions, get its bounding rectangle to crop.
13. Crop and Publish: we made the bounding rectangle bigger to have a wider view of the sign by multiplying each rectangle dimension by a factor. Then just crop the image via numpy indexing around the wider bounding rectangle and publish the image to the classifier.

For example:

![imagen](https://user-images.githubusercontent.com/67598380/178172916-aa5769ad-83d3-47f7-89bf-57885c697724.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172922-c3709d1d-86f7-42f4-afa6-69a16b798702.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172930-f0edea54-8895-4ac6-a969-42ae9c15a50d.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172933-189f2753-c8b2-4be3-b704-57ba166af4a2.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172941-b74d72fa-4fd6-4df7-bdd4-5ba6113328f9.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172945-f505ca04-e32f-4e40-84a4-7db20e745c1b.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172954-a893cfaa-cc61-46be-ae34-328c8620bdac.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172959-b0636b41-1c08-4f8f-9113-f98a747d1455.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172965-c91bd5d1-ae6d-4e51-bc6f-de878142a03f.png)
![imagen](https://user-images.githubusercontent.com/67598380/178172976-84dc11a2-6ba8-401c-8610-e69444a4d34b.png)

This solution considers only one sign at a time, so in the example above, it detects all signs but sequentially, not simultaneously as it iterates over all the contours found. You can add or remove certain signs by modifying thresholds and filtering, for example the HSV color range (we have it around blue colors but including a bit of red). The hardest to detect is the black and white sign as it is difficult to filter.

## CNN model and Classification

Convolutional Neural Networks in computer vision aim to receive an image, get the most important features or image traits with convolution and getting a smaller image in size (we used max Pooling and flattening) based on convolution results until having a features vector for classification. You need to create your model architecture and train it with image samples. The architecture of this solution's model:

![imagen](https://user-images.githubusercontent.com/67598380/178174766-c4857c75-8b76-4d41-859c-702c3c9fe032.png)

> Each phase is denoted by size, properties and/or parameters at the top of each

- Yellow refers to Conv2D it's the convolution.
- Red refers to ReLu activation function used.
- Light green refers to batch normalization. We used batch size of 16.
- Light blue refers to MaxPooling2D.
- Black refers to Dropout which is breaking some neuron connections.
- Dark green refers to Flattening for the features vector.
- Pink refers to Dense phase or neuron connection.
* We used softmax for the last activation

The most important thing to lookout was the tradeoff handling with the hardware and sign detection. The model was obtained with PC but it ran on Jetson Nano, so speed and results vary from hardware. Consider that this model is designed for embedded hardware. Jetson Nano's memory saturated if we used a more robust but heavier model, so we tried several models until we got convincing results. The other tradeoffs refer to the received images from the sign detection node. We focused that node to send minimal to none noise to prevent the classifier to classify non-sign objects increasing the risk of error, but as well verifying that the model sends out low probabilities with noisy images so we can filter those out.
This model has properties like:
- Memory required: 60.5 MB
- Batch size: 16
- Epochs: 30
- Image dimensions: 100x100x3

To train our model we used around 700 samples per class, each sample being the cropped sign just like we should receive it for new classification. We also considered Keras earlystopping callback to prevent the model to be overfit, meaning it works precisely with test-train images but not really with new images. If you want to add another sign, you should add the class in the dictionary but specially, add the class to your dataset with approximately the same number of samples as your other classes.

The code in *classify2* is the node in charge of classification. It receives the cropped image from the sign detector, loads the model, processes the incoming image to be the same dimensions, and gets the prediction. It publishes the prediction only if it is above a certain probability (we set it up at 0.80) and if it is not a repeated image, as it keeps on classifying the last image sent by the detector until it sends a new sign. Also be aware that sign detector sends lots of cropped images in a very small time, so the detector is faster than the classifier, so increasing the speed of movement of the robot is a risk of failure for classification, but also that's why the detector only sends one sign at a time, but still detects two or more if they are at sight.

!! Important: we encourage people to use their own CNN as the model has some problems with false positives !!

## Line Follower and Intersections

This node is in charge of processing everything happening on the floor. It detects the central line to follow but also crosswalks and intersections. It is based on contour detection and count. The processing is as follows:
1. Crop all but the lowest section. Vertical crop to see only the floor and horizontally to see only the central line and not the other side lines.
2. Get the centroid of the cropped section. Publish the centroid coordinates.
3. Grayscale and GaussianBlur.
4. Binzarization INV to get the black line as white pixels.
5. Filter by size. Same as sign detection, get closed components with stats and filter by size.
6. Get contours.
7. Find the biggest contour detected. This filters out possible noisy contours as the black line must be the biggest central contour.
8. Get the minAreaRect to have the enclosed minimal area rectangle from the biggest contour which returns the rectangle dimensions, angle and centroid. We publish the centroid and angle.
9. Check if no contours are detected, and then 3, it means a crosswalk has been detected, so publish a flag.

![imagen](https://user-images.githubusercontent.com/67598380/178182922-802e0363-77a8-4bab-9bc1-5599d38e5d89.png)

![imagen](https://user-images.githubusercontent.com/67598380/178182929-410824f4-6262-43c6-b3c0-c456210a9921.png)

![imagen](https://user-images.githubusercontent.com/67598380/178182934-7c365a42-59b3-459a-8ecb-e09654508923.png)

![imagen](https://user-images.githubusercontent.com/67598380/178182938-0caebcdd-b8d5-4e8f-8001-901e6af85c07.png)

![imagen](https://user-images.githubusercontent.com/67598380/178182952-2cd0c2f7-dee9-424b-9cf3-07ed69aa983c.png)

As it shows in the images, the green centroid is the crop centroid, and the red dot inside the red rectangle is the biggest contour rectangle centroid. All contours are shown in blue. The line follower is simply the difference between x-coordinates of both centroids multiplied by a proportional gain. Meaning that we want to reduce the difference between the line centroid and the crop centroid by adjusting angular speed proportionally to the difference. If the line is too far away, we shall have a more aggressive turn. 

We decided to make other help flags to understand what the robot is looking on the floor. We publish a flag to follow line if the robot detects only one contour several consecutive times. We publish another flag that the robot is looking at no contours if that's the case. And the crosswalk flag that is published every time the robot sees 0 contours and eventually sees 3 or more.

This node is the one that needs more testing and probably other methods for crosswalk detection as it's the one that showed the most failure rate. Another method thought of is creating another node specifically to detect crosswalks and intersection behaviours separated from line following, and maybe a solution is creating a state machine that has the transitions between line following to entering and intersection and the other way exiting the intersection to follow line again, considering crossing the intersection straight or turning left or right. That's for homework.

## Traffic Lights Detection

This node is a simple one. It's in charge of detecting traffic lights. It's based on HSV ranges and publishes if there is a certain green or red light in sight.
The processing is as follows:
1. Camera correction. Same as sign detection.
2. HSV colorspace and color range filtering. Create one mask for each color.
3. Erode each mask to reduce noise.
4. Dilate each mask to keep original light size for filtering.
5. Blob Detector. Generate blob detector for both masks and use blob properties to filter out possible noise. We used filter by min area, circularity and color being 255 (white pixels as HSV ranges filter to a binary image)
6. Check if there is a filtered blob in only one of the masks to publish the respective color of the mask.

![imagen](https://user-images.githubusercontent.com/67598380/178185068-df673286-7ff1-4d2b-a45f-bf1e38d1cc31.png)

![imagen](https://user-images.githubusercontent.com/67598380/178185079-84048f2c-41c4-4783-868f-10fefbb2f7e4.png)

![imagen](https://user-images.githubusercontent.com/67598380/178185101-df08c5b7-74b4-4ccf-b4b6-5c73a28c2efa.png)

The color publishing only occurs if the light changes, meaning it detects the present light in the traffic light always, but only publishes when that light changes color within the filtered characteristics.

This node challenges are mostly HSV ranges and noise filtering, but once you got those set, it works most effectively. You can add a yellow light just by adding another mask with respective HSV range and keep the same blob detector and processing for this new mask.

## Controller

This is the node that receives all the flags, states and information processed by the previous nodes and take action. The robot moves by publishing into the topic */cmd_vel* via Twist ROS message which specifies linear and angular velocities. Controller logic can be explained as a process of 4 phases:

![imagen](https://user-images.githubusercontent.com/67598380/178187552-300154b7-70da-44c9-a653-c65dd9c35dbf.png)
> From left to right, execution is sequential

There are a lot of callback functions to receive the subscriptions data which mostly represent states and flags. With all this information, the first 3 stages are for determining and updating the state of the robot, for it to take action in the 4th stage. In order:

- Stage 1: updated the boolean flag *line*, if it's true, the robot has seen several only-one contours continuously, so the state is updated to continue line and naturally not inside an intersection or crossing a crosswalk.
- Stage 2: when the boolean flag *zebra* is true, it means the robot has seen a crosswalk and is either entering or exiting the intersection. As the robot starts the track following a line and not in an intersection, the flag *in_crosswalk* is False. So if the robot enters a crosswalk, *zebra* flag is True and *in_crosswalk* flag must be updated. If the robot entered the intersection, *in_crosswalk* must be turned to True as it is crossing the intersection, and also while being inside the intersection, the robot keeps the *label* (class) of the sign detected before crossing as we don't want the robot to change the sign instruction midway even if it detects another sign while crossing the intersection. It also has to take the traffic light decision; it is taken when the robot detects the crosswalk. If the light is red, the robot must wait for green light, if not, it can cross (not wait for green light). As the traffic light has no yellow light, if the light changes while crossing the intersection, from green to red, the decision has already been taken so it keeps crossing even while red, because else the robot would have just stopped midway and we don't want that behaviour. So it can seem to happen that the robot ignores red lights, but ideally the traffic light should include yellow light, and the robot is to cross only when green light. Finally, if the robot raises the *zebra* flag and the robot is *in_crosswalk* or crossing the intersection, it means the robot has exited the intersection and should proceed to follow the line, so update that the robot is not anymore *in_crosswalk*, it has no need to wait for green light and light should be kept as green to avoid any red light changes detected while crossing.
- Stage 3: handling traffic lights. If in the previous stage the robot detected red light, it must wait for green, so if it's waiting, just check if the light has changed to green or it's still on red. Just stop if still on red, or update flag *wait_green* to stop waiting for green as light has now showed green.
- Stage 4: takes all the flags and states updated above to take movement decision. First, it checks if the robot is crossing the intersection including green light. If True, do not follow line as it must cross the intersection until exiting, but check out if the sign detected indicated cross right or cross straight, so cross the intersection accordingly. If the robot somehow messes the signs up, by having another sign that indicates other thing different from any crossing related action, it must stop as it seemed the safest option rather than crossing blindly. If the robot is not crossing any intersection, it first checks if the STOP sign has been detected and no contours are present, as STOP signs ideally do not appear in the middle of any street but always indicate to stop at maybe an intersection or crosswalk, meaning 0 contours must appear, so if the robot detects the STOP sign, it must stop until the floor has no contours, in the track, this is the case for the end of the track. If not crossing the intersection, and no stop sign, check if the sign detected is *no rules* (the black and white one), which can appear anywhere in the track, ideally not for crossing an intersection. If that sign is detected, it must be following a line, so we just add up to follow it faster (we thought of Germany's no speed limit highways). Finally, if none of the previous state are True, the robot should be continuing following line considering not in an intersection, so not waiting for green, so keep following the line.

The controller node was made of lots of if statements as all this process is within a while loop: while the ROS node is active (no shutdown), so flags and states update each time based on the ROS rate given.

The actions taken in stage 4 are specified as publishing in */cmd_vel* in the x-component of linear velocity for moving forward or backwards, and z-component of angular velocity for changing direction turning on its own axis:
- STOP: publish 0 in both angular and linear velocities to stop movement.
- CROSS_STRAIGHT: publish 0 for angular velocity and a maximum velocity for the linear velocity to only move forward at a constant speed.
- CROSS_RIGHT: ideally moves the robot around a circle based on the relation &omega;=1/R*v. But the robot does not really obey the relation, so we decided fixed values to approximate the turn. To turn left just change angular velocity sign. Important to mention that the flag */line* raised by the line follower node is the key to cross right and proceed to follow the line as the flag will indicate that a constant line has been found once the robot has crossed most of the turn.
- FOLLOW: line follower action. Takes the centroids coordinates information and computes the difference between x-coordinates. Multiplies that difference by -1 and a proportional gain to align the line and the robot proportionally. Publish that computation to the angular velocity. Publish a maximum velocity to the linear velocity to keep moving forward at constant speed while controlling the turn with the centroids difference. We added an angular velocity saturation for safety matters. Saturation at +0.2 and -0.2 because if the robot turns faster, Jetson Nano shuts down to avoid overcurrent. We also added a consideration for very narrow turns, indicated by the minAreaRect angle, if it's too close to 0 or 90°, the turn is narrow and we reduce linear velocity by half for it to have chance to correct the turn instead of overseeing the line and losing it, but these turns do not make an appearance in this track so it can be commented out.

This is the node that makes the robot move, and this solution just considers a way of updating that movement. Other methods and logic could be applied to make the same movements.

## Performance

We made several tryouts to evaluate performance. We decided to make two versions of the track for this. The first track is just the predefined track showed at the beginning, and the second track is a variation of it with more signals and higher speeds to test the limits of the solution.

With the predefined track, we got these results:

![imagen](https://user-images.githubusercontent.com/67598380/178194986-cc0267d0-995c-4b6a-967f-44159b70a907.png)

These presented a certain consistency, finishing the track successfully 7 out of 11 tries.

The second version of the track was:

![imagen](https://user-images.githubusercontent.com/67598380/178195111-83c46aea-2cf3-4ed7-b210-7f8b3de35d38.png)

And we got these results:

![imagen](https://user-images.githubusercontent.com/67598380/178195138-9daefda8-8ba8-4b68-837f-65cfe12c2b54.png)

Generally speaking, we got good results for the sign detector as failed few times and only for the detection of the black and white sign, sending only 1 noisy image sometimes. The CNN model could be better as it did well in the classification of the used signs, but when used similar signs like the roundabout sign, the predictions were very high (0.90 or above) which should have not happened as that sign is not part of a class and should be considered noise, but this should be addressed again by tradeoffs between sign detection and classification. 

The traffic lights detector showed basically no problems, only be aware about that missing yellow light. The problems came with the line follower node. Line following as an action did not have troubles at all, but the crosswalk and intersection detections did and several times. The main problem is that crosswalk detection, specifically *zebra* flag failed to be raised accordingly. Sometimes it repeated twice while it should have been raised only once, indicating the robot to follow line while crossing the intersection. Or sometimes it was not raised at all again leading to undesired behaviour. The main problem is that track cannot be finished if any intersection crossing is wrong due to detection. So we must take more testing and change intersection detection and behaviour to get better results.

## Final Comments 

This solution has several videos, presentations and more detailed reports, some in English and most in Spanish. Feel free to contact us if you need more information.

## Acknowledgments

We would like to thank the entire team in @Manchester-Robotics for all their knowledge and guidance through this great period of learning. 
We also want to thank all our teachers involved in the project, for their time, knowledge and patience. It's been really satisfying and we have learned a lot. 
