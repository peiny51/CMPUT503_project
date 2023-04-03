#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32, Int32
from turbojpeg import TurboJPEG
import cv2
import time
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
ENGLISH = False
STOP_FLAG = False

# stage 1
# 48 - Right
# 50 - Left
# 56 - Straight

# stage 2
# 163 - Duckwalk

# stage 3
# 207 - parking 1
# 226 - parking 2
# 228 - parking 3
# 75   - parking 4
# 227 - parking entrance  detect that to do stage 3



class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        name = os.environ['VEHICLE_NAME']

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + name + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + name + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher("/" + name + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.id_sub = rospy.Subscriber("/" + name + "/april_tag_id",
                                    Int32,
                                    self.callback_at,
                                    queue_size=1)
        
               
        

        self.jpeg = TurboJPEG()
        self.id_num = 0 
        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -240
        else:
            self.offset = 240
        self.velocity = 0.26
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.twist1 = Twist2DStamped(v=0, omega=0)
        self.P = 0.04
        self.D = -0.004
        self.I = 0.008
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def callback_at(self, msg):
        self.id_num = msg.data 
        # rospy.loginfo(f'>> Tag ID detected in lane following: {self.id_num}')

    def controller(self): 
        # rospy.loginfo(f'## Tag ID detected in controller: {self.id_num}')
        if self.id_num == 9999:
            rospy.signal_shutdown("Done!")
        if self.id_num == 1000:
            print("stopping")
            self.stop(3)
            self.id_num = 0
            self.velocity = 0.26
        if self.id_num == 2000:
            print("stopping before ducks")
            # self.stop(3)
            self.stop_until()
            self.velocity = 0.0
        elif self.id_num == 56:
            self.stop(3)
            self.straight(self.id_num)
            self.id_num = 0
        elif self.id_num == 48:
            print("right")              
            self.stop(3)
            self.right(self.id_num)
            self.id_num = 0
        elif self.id_num == 50:
            print("left")               
            self.stop(3)
            self.left(self.id_num)
            self.id_num = 0
        elif self.id_num == 1:
            print("go stright")               
            self.straight(self.id_num)
            self.id_num = 0

        else: 
            print("driving2")
            self.drive()
            
            
    def stop(self, tm):
        self.twist1.v = 0
        self.twist1.omega = 0
        self.vel_pub.publish(self.twist1)
        rospy.sleep(tm)
        
    def stop_until(self):
        self.twist1.v = 0
        self.twist1.omega = 0
        self.vel_pub.publish(self.twist1)
        
        # self.proportional = None

    
    def straight(self, id_num):
        # print("straight") 
        self.twist1.v = 0.3
        self.twist1.omega = 0
        if id_num == 56:
            self.twist1.omega = -0.1
        else:
            self.twist1.omega = -0.09
            
        rate = rospy.Rate(5)
        for i in range(15):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()

        self.last_error = 0
        # self.proportional = None
    

    def right(self, id_num):
        loop = 8
        if id_num == 48:
            self.twist1.v = 0.2
            self.twist1.omega = -1
        rate = rospy.Rate(5)
        for i in range(loop):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()
        self.last_error = 0
        # self.proportional = None
        
    def left(self, id_num):
        loop = 8
        if id_num == 50:
            self.twist1.v = 0.2
            self.twist1.omega = 1
            loop = 15
        rate = rospy.Rate(5)
        for i in range(loop):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()
        self.last_error = 0
        # self.proportional = None


    def drive(self):
        if self.proportional is None:
            self.twist1.omega = 0
            self.last_error = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_time = (rospy.get_time() - self.last_time)
            d_error = (self.proportional - self.last_error) / d_time
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            # I Term
            I = -self.proportional * self.I * d_time

            self.twist.v = self.velocity
            self.twist.omega = P + I + D
            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist1.v = 0
        self.twist1.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)
            
       

if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.controller()
        rate.sleep()
