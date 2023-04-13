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
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, WheelEncoderStamped
from math import pi

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
        self.host = str(os.environ['VEHICLE_NAME'])
        
        self.reset()

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
        
        self.right_tick_sub = rospy.Subscriber(f'/{name}/right_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.right_tick,  queue_size = 1)
        self.left_tick_sub = rospy.Subscriber(f'/{name}/left_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.left_tick,  queue_size = 1)
        
        self.sub_detection = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, self.cb_detection, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.host), Float32, self.cb_distance, queue_size=1)
        
        self.r = rospy.get_param(f'/{name}/kinematics_node/radius', 100)
        
        self.pub_parking_detection = rospy.Publisher("/" + name + "/parking_detection", BoolStamped,queue_size=1)

        self.jpeg = TurboJPEG()
        self.id_num = 0 
        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -240
        else:
            self.offset = 240
        self.velocity = 0.23
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.twist1 = Twist2DStamped(v=0, omega=0)
        self.P = 0.04
        self.D = -0.004
        self.I = 0.008
        self.last_error = 0
        self.safe_dist = 50
        self.last_time = rospy.get_time()
        self.l = rospy.get_time()
        self.check = False
        self.first = True
        self.bias = 0 # -0.09
        self.eng = False

        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # Shutdown hook
        rospy.on_shutdown(self.hook)
        

    def reset(self):
        self.rt_initial_set = False
        self.rt = 0
        self.rt_initial_val = 0

        self.lt_initial_set = False
        self.lt = 0
        self.lt_initial_val = 0
        
        self.prv_rt = 0
        self.prv_lt = 0
        
        self.Th = 0
        self.L = 0.05


    def right_tick(self, msg):
        if not self.rt_initial_set:
            self.rt_initial_set = True
            self.rt_initial_val = msg.data
        self.rt = msg.data - self.rt_initial_val
        # self.rt_dist = (2 * pi * self.r * self.rt_val) / 135


    def left_tick(self, msg):
        if not self.lt_initial_set:
            self.lt_initial_set = True
            self.lt_initial_val = msg.data
        self.lt = msg.data - self.lt_initial_val
        # self.lt_dist = (2 * pi * self.r * self.lt_val) / 135
        
    def task_rotation(self, rotation_amount, omega_amount, stopping_offset):
        rospy.loginfo("Starting rotation task..")
        rospy.loginfo(f'Initial Theta: {self.Th}')
        msg_velocity = Twist2DStamped()
        rate = rospy.Rate(10)

        initial_theta = self.Th
        
        target_theta = initial_theta + rotation_amount
        prv_rt = self.rt
        prv_lt = self.lt

        while True:
            delta_rt = self.rt - prv_rt 
            delta_lt = self.lt - prv_lt
            prv_rt = self.rt
            prv_lt = self.lt
        
            delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
            delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

            delta_dist_cover = (delta_rw_dist + delta_lw_dist)/2

            self.Th  += (delta_rw_dist - delta_lw_dist) / (2 * self.L)
        
            if rotation_amount < 0 and self.Th - stopping_offset < target_theta:
                break
            if rotation_amount > 0 and self.Th + stopping_offset > target_theta:
                break
 
            self.twist1.v = 0.15
            self.twist1.omega = omega_amount

            self.vel_pub.publish(self.twist1)
            # rospy.loginfo(f'Self.Th: {self.Th}, target_th: {target_theta}')
            rate.sleep()

        # rospy.loginfo(f'Final Theta: {self.Th}')
    
    def cb_detection(self, bool_stamped):
        """
        call back function for leader detection
        """
        self.detection = bool_stamped.data 
        # if self.detection:
        #     self.velocity = 0.2
            
            
    def cb_distance(self, distance):
        """
        call back function for leader distance
        """
        
        # self.velocity = 0.15
        
        self.distance = 100 * (distance.data)
        # rospy.loginfo(f'Distance from the robot in front: {self.distance}')
        
        if self.distance < self.safe_dist and self.first == True:
            self.first = False
            rospy.loginfo('Eng')
            self.id_num = 500
            self.eng = True
            # self.stop(3)
            # self.id_num = 0
            # self.offset = -160
            

        
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
                
                if self.offset < 240 and self.id_num == 0:
                    # rospy.loginfo("hehe")
                    if self.check == False:
                        self.l = rospy.get_time()
                        self.prv_rt = self.rt
                        self.prv_lt = self.lt
                    duration = (rospy.get_time() - self.l)
                    self.bais = 0
                    
                    delta_rt = self.rt - self.prv_rt 
                    delta_lt = self.lt - self.prv_lt
                
                    delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
                    delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

                    delta_dist_cover = (delta_rw_dist + delta_lw_dist)/2
                    
                    delta_dist_cover = int(delta_dist_cover * 100) + 1
                    
                    # rospy.loginfo(f'Dist: {delta_dist_cover}')
                    
                    self.velocity = 0.2
                   
                    # if delta_dist_cover >= 0 and delta_dist_cover < 30:
                    #     self.velocity = 0.2
                        
                    #     self.offset = -90 -  (delta_dist_cover * 1)
                   
                    if delta_dist_cover > 25:  #
                        
                        # self.task_rotation(-pi/16, -6, 0)
                        # self.task_rotation(-pi/8, -3, 0)
                        # self.task_rotation(-pi/20, -6, 0)
                        # self.task_rotation(-pi/6, -3, 0)
                        # self.straight(501)
                        # self.last_error = 0
                        
                        self.id_num = 999
                        
                        
                        rospy.loginfo("4 seconds")
                        self.check = False
                        self.first = True
                        # self.bais = -0.09
                        self.last_error = 0
                        self.eng = False
                        
                        self.Th = 0
                        self.task_rotation(-pi/9, -5, 0)
                        self.stop(0.1)
                        self.straight(600)
                        self.stop(0.1)
                        prv_t = -self.Th
                        self.Th = 0
                        self.task_rotation(prv_t, 5, 0)
                        self.stop(0.1)
                        
                        self.offset = 240
                        
                        self.id_num = 0
                        
                    self.check = True
                    
                self.proportional = cx - int(crop_width / 2) + self.offset
                
                # rospy.loginfo(f'offset: {self.offset}; proportional: {self.proportional}')
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
        if self.id_num == 999:
            return
        if self.id_num == 500:
            print("stopping")
            self.stop(2)
            self.Th = 0
            self.task_rotation(pi/9, 5, 0)
            self.stop(0.1)
            self.straight(600)
            self.stop(0.1)
            prv_t = self.Th
            self.Th = 0
            self.task_rotation(-prv_t, -5, 0)
            self.stop(0.1)
            # self.task_rotation(pi/10, 3, 0)
            # self.straight(500)
            
            self.last_error = 0
            self.id_num = 0
            self.velocity = 0.23
            self.offset = -210

            # self.stop(0.5)
        if self.id_num == 1000:
            print("stopping")
            self.stop(3)
            self.id_num = 0
            self.velocity = 0.23
        if self.id_num == 2000:
            print("stopping before ducks")
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
            self.stop(2)
            print("go stright")               
            self.straight2(self.id_num)
            self.velocity = 0.23
            self.id_num = 0
        elif self.id_num == 38:
            # parking
            self.stop(2)
            print("parking")               
            # self.straight2(self.id_num)
            self.velocity = 0.23
            parking_detection_flag = BoolStamped()
            parking_detection_flag.data = True
            self.pub_parking_detection.publish(parking_detection_flag)
            self.id_num = 39
            return
        elif self.id_num == 39:
            self.velocity = 0
            self.twist1.v = 0
            self.twist1.omega = 0
            return
        else: 
            # print("driving2")
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
        
        self.proportional = None

    
    def straight(self, id_num):
        # print("straight") 
        self.twist1.v = 0.23
        self.twist1.omega = 0
        
        loop = 15
        if id_num == 56: ## STAGE 1 STRAIGHT
            prv_rt = self.rt
            prv_lt = self.lt
            
            distance = 0
            
            r = rospy.Rate(15)
            
            while distance < 0.42:
                delta_rt = self.rt - prv_rt 
                delta_lt = self.lt - prv_lt
                
                delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
                delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

                distance = (delta_rw_dist + delta_lw_dist)/2
                
                self.twist1.v = 0.23
                self.twist1.omega = 0.2
                
                self.vel_pub.publish(self.twist1)
                r.sleep()
            return
        elif id_num == 600:
            
            prv_rt = self.rt
            prv_lt = self.lt
            
            distance = 0
            
            r = rospy.Rate(15)
            
            while distance < 0.23:
                delta_rt = self.rt - prv_rt 
                delta_lt = self.lt - prv_lt
                
                delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
                delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

                distance = (delta_rw_dist + delta_lw_dist)/2
                
                self.twist1.v = 0.23
                self.twist1.omega = -0.09
                
                self.vel_pub.publish(self.twist1)
                r.sleep()
            return
        elif id_num == 501:
            self.twist1.omega = 1.3
            loop = 10
        elif id_num == 111:
            self.twist1.omega = -0.09
            loop = 5
        else:
            self.twist1.omega = -0.09
            
        rate = rospy.Rate(5)
        for i in range(loop):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()

        # self.last_error = 0
        self.proportional = None
    
    def straight2(self, id_num):
        # print("straight") 
        self.twist1.v = 0.3
        self.twist1.omega = 0
        if id_num == 56:
            self.twist1.omega = -0.1
        else:
            self.twist1.omega = -0.09
        
        
        rate = rospy.Rate(5)
        for i in range(10):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()



    def right(self, id_num):
        loop = 8
        if id_num == 48:
            prv_rt = self.rt
            prv_lt = self.lt
            
            distance = 0
            
            r = rospy.Rate(15)
            # go straight for some distance
            while distance < 0.06:
                delta_rt = self.rt - prv_rt 
                delta_lt = self.lt - prv_lt
                
                delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
                delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

                distance = (delta_rw_dist + delta_lw_dist)/2
                
                self.twist1.v = 0.15
                self.twist1.omega = -0.09
                
                self.vel_pub.publish(self.twist1)
                r.sleep()
            
            self.twist1.v = 0.2
            self.twist1.omega = -1   #-0.4
            
            
        if id_num == 500:
            self.twist1.v = 0.05
            self.twist1.omega = -2.6
            loop = 7
        rate = rospy.Rate(5)
        for i in range(loop):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()
        # self.last_error = 0
        self.proportional = None
        
    def left(self, id_num):
        loop = 8
        if id_num == 50:
            self.twist1.v = 0.24
            self.twist1.omega = 1.8
            loop = 12
        if id_num == 500:
            self.twist1.v = 0.05
            self.twist1.omega = 4
            loop = 7
        rate = rospy.Rate(5)
        for i in range(loop):     
            self.vel_pub.publish(self.twist1)
            rate.sleep()
        # self.last_error = 0
        self.proportional = None


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
            self.twist.omega = P + I + D + self.bias
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
