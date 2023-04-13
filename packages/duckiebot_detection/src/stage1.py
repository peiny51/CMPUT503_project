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
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection, Twist2DStamped, BoolStamped
from duckietown_msgs.srv import GetVariable

import rospy
import time
import rospkg

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Transform, Vector3, Quaternion
from std_msgs.msg import String, Float32, Int32

from math import pi

LEFT = 50
RIGHT = 48
STOP = 1000 
STOP_UNTIL = 2000

# 48 - Right
# 50 - Left
# 56 - Straight
# 163 - Duckwalk  stop sign
# 207 - parking 1
# 226 - parking 2
# 228 - parking 3
# 75   - parking 4
# 227 - parking entrance

class TagDetector(DTROS):
    def __init__(self, node_name):
        super(TagDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        name = os.environ['VEHICLE_NAME']
        self.image_sub = rospy.Subscriber(f'/{name}/camera_node/image/compressed', CompressedImage, self.rcv_img,  queue_size = 1)  
        self.image_pub = rospy.Publisher(f'/{name}/digit_detector_node/image/compressed', CompressedImage,  queue_size = 1) 
        self.id_pub = rospy.Publisher("/" + name + "/april_tag_id", Int32, queue_size=1)
        
        
        
        
        
        self.img_queue = deque(maxlen=1)
        self.rectify_alpha = rospy.get_param("~rectify_alpha", 0.0)
        self.jpeg = TurboJPEG()
        self._map_xy_set = False
        self.bridge = CvBridge() 
        
        yaml_path = f'/data/config/calibrations/camera_intrinsic/{name}.yaml'
        self.init_undistor_img(yaml_path)

        self.family = rospy.get_param("~family", "tag36h11") 
        self.nthreads = rospy.get_param("~nthreads", 1)
        self.quad_decimate = rospy.get_param("~quad_decimate", 1.0)
        self.quad_sigma = rospy.get_param("~quad_sigma", 0.0)
        self.refine_edges = rospy.get_param("~refine_edges", 1)
        self.decode_sharpening = rospy.get_param("~decode_sharpening", 0.25)
        self.tag_size = rospy.get_param("~tag_size", 0.065)
        
        

        # self.tag_id_uofa = [93, 94, 200, 201]  # very outer 
        self.tag_id_lt = [50]
        self.tag_id_rt = [48]
        self.tag_id_stop = [163, 38, 21]   # 21 is in grad room
        self.turning_tags = [48, 50, 56, 163, 21, 38]
        
        self.img_queue = deque(maxlen=1)

        self.detector =  Detector(
            families=self.family,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,) 
        

        self.dist_threshold = 0.6  # 0.6
        self.next_intersection = -1
        self.stage3_done = False 
        # self.d = 100
        self.counter = 0
        self.b = 0
        self.d = 0
        
        
        
        
    def rcv_img(self, msg):
        image = self.jpeg.decode(msg.data)
        self.img_queue.append(image)
        
    
    def run(self): 
        rate = rospy.Rate(5) # 5Hz== 10
        while not rospy.is_shutdown():
            # if self.next_intersection == 1234:
            #     self.id_pub.publish(9999)
            #     rospy.signal_shutdown("Done")
            if self.img_queue:
                img = self.img_queue.popleft()  
                self.process_img(img)
            rate.sleep()
    
    
    def process_img(self, img):
        # rospy.loginfo(f'd: {self.d}')
        min_dist = -1
        target_size = self.detect_intersection(img)
        
        blue_ahead = self.detect_crossroad(img)
        self.b = blue_ahead + self.b
        duck_ahead = self.detect_ducks2(img)
        self.d = duck_ahead + self.d
        
        if self.counter % 2 == 0:
            blue_ahead = self.b  / 2
            duck_ahead = self.d / 2 
        
            # rospy.loginfo(f'target size1: {target_size2}')
                
            if target_size > 0.2 and self.next_intersection != -1:
                # rospy.loginfo(f'At intersection, target size: {target_size}')
                self.id_pub.publish(self.next_intersection)
                self.next_intersection = -1
            elif blue_ahead > 0.2 and self.next_intersection == 163:  # 0.15
                # rospy.loginfo('stops')
                self.id_pub.publish(STOP_UNTIL)
                # rospy.loginfo('stops2')
                if duck_ahead < 0.07:
                    # rospy.loginfo('begin publish 1')
                    self.id_pub.publish(1)  # driving
                    # rospy.loginfo('after publish 1')
                    self.next_intersection = -1
                    # self.d = 100
                
            else:
                undistorted_img = self.undistort_img(img)
                tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size)                

                
                min_tag_id = -1 
                min_tag = None
                
                for tag in tags:
                    p = tag.pose_t.T[0]
                    tag_id=int(tag.tag_id)
                    dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

                    if min_dist == -1:
                        min_dist = dist
                        min_tag_id = tag_id 
                        min_tag = tag
                        
                    elif dist < min_dist:
                        min_dist = dist
                        min_tag_id = tag_id 
                        min_tag = tag

                if min_tag_id != -1: # We next_interhave the tag with minimum distance
                    # self.d = min_dist
                    if self.next_intersection == -1 and min_tag_id in self.turning_tags:
                        self.next_intersection = min_tag_id
                        rospy.loginfo(f'Intersection set to: {min_tag_id}')
                            
            self.b = 0
            self.d = 0        
        self.counter = self.counter + 1
                
    def detect_ducks2(self, img):
        h, w, d = img.shape
        crop = img[300:-1, 0:int(0.65*w), :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([0,50,0])
        upper_orange = np.array([50,255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # # # # mask = mask0 + mask1
        # # # # target_size = np.sum(mask/255.) / mask.size
        
        # mask_or = cv2.bitwise_or(mask0, mask1)
        # target_size1 = np.sum(mask_or/255.) / mask_or.size
        
        target_size = np.sum(mask/255.) / mask.size
        # rospy.loginfo(f'orange mask:{target_size}')
        return target_size
        # if target_size > 0.1:
        #     return True
        # else:
        #     return False


    def detect_ducks(self, img, dist):
        rospack = rospkg.RosPack()
        path = rospack.get_path("duckiebot_detection") 
        img_path = str(path)+"/src/close_crop.png"
        template = cv2.imread(img_path)
        
        detect = False
        DEBUG = True
        
        # h, w, d = img.shape
        
        # x1 = int((0.25 + (1-dist)/8)*h)
        # x2 = int((1-dist)*h)
        # cropped_img = img[x1:x2, 0:int(0.65*w),:]
        

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = .8
        loc = np.where(res >= threshold)

        if len(loc[0]) != 0:
            detect = True
        else:
            detect =  False
        
        # if DEBUG:
        #     rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        #     self.image_pub.publish(rect_img_msg)
        
        # rospy.loginfo(f"detect: {detect}")
        return True
        
    def detect_intersection(self, img):
        crop = img[360:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # join my masks
        mask = mask0+mask1
        target_size = np.sum(mask/255.) / mask.size
        return target_size
    
    
    def detect_crossroad(self, img):
        h, w, d = img.shape
        crop = img[300:-1, 0:int(0.65*w), :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([80,100,0])
        upper_blue = np.array([130,255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # result = cv2.bitwise_and(crop, crop, mask = mask)
        target_size = np.sum(mask/255.) / mask.size

        
        # rospy.loginfo(f'blue mask:{target_size}')
        return target_size
        # if target_size > 0.2:
        #     return True
        # else:
        #     return False


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
    tag_detector = TagDetector(node_name='tag_detector')
    tag_detector.run()
    rospy.spin()
