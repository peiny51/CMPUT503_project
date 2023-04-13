#!/usr/bin/env python3

import os
import numpy as np
import cv2
import yaml
import tf
import math
import numpy as np
from cv_bridge import CvBridge
from turbojpeg import TurboJPEG
from collections import deque
from turbojpeg import TurboJPEG, TJPF_GRAY
from dt_apriltags import Detector
from duckietown_msgs.msg import WheelEncoderStamped, Twist2DStamped, BoolStamped
from duckietown_msgs.srv import GetVariable

import rospy
import time

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Transform, Vector3, Quaternion
from std_msgs.msg import String, Float32, Int32

from math import pi


class ParkingNode(DTROS):
    def __init__(self, node_name):
        super(ParkingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        name = os.environ['VEHICLE_NAME']
        self.host = str(os.environ['VEHICLE_NAME'])
        self.start = False
        
        self.image_sub = rospy.Subscriber(f'/{name}/camera_node/image/compressed', CompressedImage, self.rcv_img,  queue_size = 1)   
        self.vel_pub = rospy.Publisher("/" + name + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        self.pub_parking_detection = rospy.Subscriber("/" + name + "/parking_detection", BoolStamped, self.cb_parking_detection, queue_size=1)
        
        self.reset()

        self.right_tick_sub = rospy.Subscriber(f'/{name}/right_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.right_tick,  queue_size = 1)
        self.left_tick_sub = rospy.Subscriber(f'/{name}/left_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.left_tick,  queue_size = 1)
        
        self.img_queue = deque(maxlen=1)
        self.rectify_alpha = rospy.get_param("~rectify_alpha", 0.0)
        self.jpeg = TurboJPEG()
        self._map_xy_set = False
        self.bridge = CvBridge() 
        
        self.r = rospy.get_param(f'/{name}/kinematics_node/radius', 100)
        
        yaml_path = f'/data/config/calibrations/camera_intrinsic/{name}.yaml'
        self.init_undistor_img(yaml_path)

        self.family = rospy.get_param("~family", "tag36h11") 
        self.nthreads = rospy.get_param("~nthreads", 1)
        self.quad_decimate = rospy.get_param("~quad_decimate", 1.0)
        self.quad_sigma = rospy.get_param("~quad_sigma", 0.0)
        self.refine_edges = rospy.get_param("~refine_edges", 1)
        self.decode_sharpening = rospy.get_param("~decode_sharpening", 0.25)
        self.tag_size = rospy.get_param("~tag_size", 0.065)
        
        self.img_queue = deque(maxlen=1)

        self.detector =  Detector(
            families=self.family,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,) 
        

        self.parking_spot = 4
        self.state = "STRAIGHT"
        self.turns = ["D", "Turn1", "Turn2", "Turn3", "Turn4"]
        self.turn_ids = [0, 207, 226, 228, 75]
        self.turn_omega = [0, 3.5, 3.5, -3.1, -3]
        self.center_bias = [15, 30, 0, -12, -40]
        self.time_detect = [0.5, 0.55, 0.55, 0.45, 0.45]
        self.time_not_detect = [0.5, 0.65, 0.65, 0.5, 0.5]

        # PID Variables
        self.proportional = None
        self.omega_bias = -0.095
        self.offset = 240
        self.velocity = 0
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.P = 0.04
        self.D = -0.004
        self.I = 0.008
        self.last_error = 0
        self.last_time = rospy.get_time()

    def reset(self):
        self.rt_initial_set = False
        self.rt = 0
        self.rt_initial_val = 0

        self.lt_initial_set = False
        self.lt = 0
        self.lt_initial_val = 0
        
        self.prv_rt = 0
        self.prv_lt = 0
        
        self.vel_incr = 0


    def right_tick(self, msg):
        if not self.rt_initial_set:
            self.rt_initial_set = True
            self.rt_initial_val = msg.data
        self.rt = msg.data - self.rt_initial_val


    def left_tick(self, msg):
        if not self.lt_initial_set:
            self.lt_initial_set = True
            self.lt_initial_val = msg.data
        self.lt = msg.data - self.lt_initial_val

    def cb_parking_detection(self, msg):
        if msg.data:
            self.start = True
        
        
    def rcv_img(self, msg):
        image = self.jpeg.decode(msg.data)
        self.img_queue.append(image)
    
    def run(self): 
        rate = rospy.Rate(5) # 5Hz== 10
        while not rospy.is_shutdown():
            if self.start and self.img_queue:
                img = self.img_queue.popleft() 
                undistorted_img = self.undistort_img(img)
                
                delta_rt = self.rt - self.prv_rt
                delta_lt = self.lt - self.prv_lt

                self.prv_rt = self.rt
                self.prv_lt = self.lt

                delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
                delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

                delta_dist_cover = (delta_rw_dist + delta_lw_dist)/2
                
                if self.state == "STRAIGHT":
                    tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size) 
                    str_id = int(227)
                    str_tag = None
                    for tag in tags:
                        if str_id == int(tag.tag_id):
                            str_tag = tag
                            break

                    if str_tag is not None:
                        # rospy.loginfo("Tag 227 detected.")
                        p = str_tag.pose_t.T[0]
                        dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

                        stop_dist = 0.17
                        if self.parking_spot == 3:
                            stop_dist = 0.185
                        if self.parking_spot == 2:
                            stop_dist = 0.37
                        if self.parking_spot == 4:
                            stop_dist = 0.37
                            

                        if dist < stop_dist:
                            self.state = self.turns[self.parking_spot]
                            self.twist.v = 0
                            self.twist.omega = 0
                            self.vel_pub.publish(self.twist)
                            continue
                        else: 
                            if delta_dist_cover == 0:
                                self.vel_incr += 0.06 
                            else:
                                self.vel_incr = 0
                                
                            self.velocity = 0.18 + self.vel_incr
                                
                            
                        center=str_tag.center.tolist()[0] - self.center_bias[0]
                        crop_width = undistorted_img.shape[1]
                        self.proportional = ((center) - int(crop_width / 2)) / 5
                    else:
                        self.velocity = 0

                    self.drive()

                if self.state == "Turn1" or self.state == "Turn2" or self.state == "Turn3" or self.state == "Turn4":
                    tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size) 
                    turn_id = int(self.turn_ids[self.parking_spot])
                    rospy.loginfo(f'Turn id: {turn_id}')

                    turn_tag = None
                    for tag in tags:
                        rospy.loginfo(f'Detect id: {tag.tag_id}')
                        if turn_id == int(tag.tag_id):
                            turn_tag = tag
                            break
                    if turn_tag is not None:
                        center=turn_tag.center.tolist()[0]
                        rospy.loginfo(f'Center at: {center}')
                        if center < 290:
                            self.twist.v = 0
                            self.twist.omega = abs(self.turn_omega[self.parking_spot]) * 1.8
                            if turn_id == self.turn_ids[3] or turn_id == self.turn_ids[4]:
                                self.twist.omega = abs(self.turn_omega[self.parking_spot]) * 1.35
                            self.vel_pub.publish(self.twist)
                            time.sleep(self.time_detect[self.parking_spot])
                            self.twist.v = 0
                            self.twist.omega = 0
                            self.vel_pub.publish(self.twist) 
                            time.sleep(0.12)
                        elif center > 350:
                            self.twist.v = 0
                            self.twist.omega = -abs(self.turn_omega[self.parking_spot]) * 1.8
                            if turn_id == self.turn_ids[3] or turn_id == self.turn_ids[4]:
                                self.twist.omega = -abs(self.turn_omega[self.parking_spot]) * 1.35
                            self.vel_pub.publish(self.twist)
                            time.sleep(self.time_detect[self.parking_spot])
                            self.twist.v = 0
                            self.twist.omega = 0
                            self.vel_pub.publish(self.twist) 
                            time.sleep(0.12)
                        else:
                            self.twist.v = 0
                            self.twist.omega = 0
                            self.vel_pub.publish(self.twist) 
                            self.last_error = 0
                            self.state = "Park"
                    else:
                        self.twist.v = 0
                        self.twist.omega = self.turn_omega[self.parking_spot]
                        self.vel_pub.publish(self.twist)
                        time.sleep(self.time_not_detect[self.parking_spot])
                        self.twist.v = 0
                        self.twist.omega = 0
                        self.vel_pub.publish(self.twist)
                        time.sleep(0.12) 

                if self.state == "Park":
                    tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size) 
                    str_id = int(self.turn_ids[self.parking_spot])
                    str_tag = None
                    for tag in tags:
                        if str_id == int(tag.tag_id):
                            str_tag = tag
                            break

                    if str_tag is not None:
                        # rospy.loginfo("Tag 227 detected.")
                        p = str_tag.pose_t.T[0]
                        dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

                        stop_dist = 0.19
                        
                        if dist < stop_dist:
                            self.twist.v = 0
                            self.twist.omega = 0
                            self.vel_pub.publish(self.twist)
                            time.sleep(0.12)
                            self.state = "End"
                            self.velocity = 0
                            self.vel_incr = 0
                            # rospy.signal_shutdown("Done!")
                            continue
                        else: 
                            if self.parking_spot == 1:
                                self.velocity = 0.17
                            elif self.parking_spot == 2:
                                self.velocity = 0.15
                            else:
                                self.velocity = 0.16
                                
                            if delta_dist_cover == 0:
                                self.vel_incr += 0.06
                            else:
                                self.vel_incr = 0
                                
                            self.velocity = self.velocity + self.vel_incr
                            
                        center=str_tag.center.tolist()[0] - self.center_bias[self.parking_spot]
                        crop_width = undistorted_img.shape[1]
                        self.proportional = ((center) - int(crop_width / 2)) / 3.5
                    else:
                        self.velocity = 0

                    self.drive()


            rate.sleep()     

    def drive(self):
        if self.proportional is None:
            self.twist.omega = 0
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
            self.twist.omega = P + I + D + self.omega_bias
            
            # self.loginfo(f'Prop: {self.proportional}, Omg: {self.twist.omega}, Vel: {self.twist.v}')

        self.vel_pub.publish(self.twist)          


    def undistort_img(self, img):
        if self._map_xy_set:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)
            return img
        else:
            return None


    def init_undistor_img(self, filename):
        # https://github.com/duckietown/dt-core/blob/4245513ad962ab7a880030a7a5ad2f43dca8b076/packages/apriltag/src/apriltag_detector_node.py#L188
        cam_info = self.load_camera_info(filename)
        H, W = cam_info.height, cam_info.width
        K = np.array(cam_info.K).reshape(3,3)
        D = np.array(cam_info.D)
        rect_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), self.rectify_alpha)
        
        self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])
        
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(K, D, None, rect_K, (W, H), cv2.CV_32FC1)

        self._map_xy_set = True
        # rospy.loginfo(f'map_x: {self._mapx} \n\n map_y: {self._mapy}')
        

    def load_camera_info(self, filename: str) -> CameraInfo:
        # https://github.com/duckietown/dt-core/blob/4245513ad962ab7a880030a7a5ad2f43dca8b076/packages/complete_image_pipeline/include/image_processing/calibration_utils.py
        
        with open(filename, "r") as f:
            calib_data = yaml.load(f, Loader=yaml.Loader)
        cam_info = CameraInfo()
        cam_info.width = calib_data["image_width"]
        cam_info.height = calib_data["image_height"]
        cam_info.K = calib_data["camera_matrix"]["data"]
        cam_info.D = calib_data["distortion_coefficients"]["data"]
        cam_info.R = calib_data["rectification_matrix"]["data"]
        cam_info.P = calib_data["projection_matrix"]["data"]
        cam_info.distortion_model = calib_data["distortion_model"]
        return cam_info

if __name__ == '__main__': 
    parking_node = ParkingNode(node_name='parking_node')
    parking_node.run()
    rospy.spin()
