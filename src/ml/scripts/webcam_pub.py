#!/usr/bin/env python3

import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2   

def publish_message():
 
  
  pub = rospy.Publisher('video_frames', Image, queue_size=10)
     
  rospy.init_node('video_pub_py', anonymous=True)
     
  
  rate = rospy.Rate(10) # 10hz
     
  
  # cap = cv2.VideoCapture(0)
     
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # While ROS is still running.
  while not rospy.is_shutdown():
     
      # Capture frame-by-frame
      # This method returns True/False as well
      # as the video frame.
      # ret, frame = cap.read()
      frame = cv2.imread("/home/ahmed/catkin_ws/src/ml/scripts/IMG_1159.JPG")
      print(frame.shape)

      ret = True
      if ret == True:
        # Print debugging information to the terminal
        rospy.loginfo('publishing video frame')

        pub.publish(br.cv2_to_imgmsg(frame))
             
      # Sleep just enough to maintain the desired rate
      rate.sleep()
         
if __name__ == '__main__':
  try:
    publish_message()
  except rospy.ROSInterruptException:
    pass