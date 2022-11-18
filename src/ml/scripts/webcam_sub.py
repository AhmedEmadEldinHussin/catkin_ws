#!/usr/bin/env python3

import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import tensorflow as tf
import numpy as np
from model import build_model
from std_msgs.msg import String

model = build_model((128,128,3))
class_names = ['0','1','2','3']
cat

def preprocess(image):
  img  = image.copy()
  img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  img  = cv2.resize(img,(128,128))
  img = np.expand_dims(img,axis=0)
  return img 

def do_prediction(image):
  processed_image = preprocess(image)
  predictions = model.predict(processed_image)
  cls = np.argmax(predictions)
  score = tf.nn.softmax(predictions[0])
  mx = tf.argmax(score)
  print(mx)
  return cls, confidence
  

 
def callback(data):
 
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # Output debugging information to the terminal
  rospy.loginfo("receiving video frame")
   
  # Convert ROS Image message to OpenCV image
  current_frame = br.imgmsg_to_cv2(data)
  cls , score = do_prediction(current_frame)
  # send_result(cls,score)
  pub = rospy.Publisher('ml_result', String, queue_size=10)
  result = f'cls: {cls} , and score: {score}'
  rospy.loginfo("publishing ml result")
  pub.publish(result)
   
  # Display image
  cv2.imshow("camera", current_frame)
   
  cv2.waitKey(1)
      
def receive_message():
 
 
  rospy.init_node('video_sub_py', anonymous=True)
   
  # Node is subscribing to the video_frames topic
  rospy.Subscriber('video_frames', Image, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()



  
if __name__ == '__main__':
  receive_message()