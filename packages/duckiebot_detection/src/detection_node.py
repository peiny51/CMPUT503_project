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
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection, Twist2DStamped
from duckietown_msgs.srv import GetVariable

import rospy
import time

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Transform, Vector3, Quaternion
from std_msgs.msg import String, Float32, Int32

LEFT1 = 153
LEFT2 = 162
RIGHT1 = 169
RIGHT2 = 58
STR1 = 143
STR2 = 62
STOP = 1000 

class TagDetector(DTROS):
    def __init__(self, node_name):
        super(TagDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        name = os.environ['VEHICLE_NAME']
        
        self.image_sub = rospy.Subscriber(f'/{name}/camera_node/image/compressed', CompressedImage, self.rcv_img,  queue_size = 1)   
        self.id_pub = rospy.Publisher("/" + name + "/april_tag_id", Int32, queue_size=1)
        self.image_pub = rospy.Publisher(f'/{name}/digit_detector_node/image/compressed', CompressedImage,  queue_size = 1)
        self.sub_status = rospy.Subscriber("/" + name + "/car_status",Float32,self.callback_status, queue_size=1)
        self.digit_sub = rospy.Subscriber(f'/{name}/digit', Int32, self.cb_digit,  queue_size = 1)
        self.status = 50000 
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

        self.tag_id_uofa = [93, 94, 200, 201]
        self.tag_id_lt = [153, 62]
        self.tag_id_rt = [133, 58]
        self.tag_id_stop = [169, 162]

        self.detections = []
        self.digit_detections = []
        self.digit = -1
        
        self.img_queue = deque(maxlen=1)

        self.detector =  Detector(
            families=self.family,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,) 
        
        self.tag_info = {
            200: [0.17, 0.17, 0.065, 1],
            201: [1.65, 0.17, 0.065, 1],
            94:  [1.65, 2.84, 0.065, 1],
            93:  [0.17, 2.84, 0.065, 1],
            153: [1.75, 1.252,0.065, 1],
            143: [1.253,1.755,0.065, 1],
            58:  [0.574,1.259,0.065, 1],
            62:  [0.075,1.755,0.065, 1],
            169: [0.574,1.755,0.065, 1],
            162: [1.253,1.253,0.065, 1]
        }

        self.turning_tags = [153, 162, 169, 58, 143, 62]

        self.dist_threshold = 0.6  # 0.6

        self.next_intersection = -1
        self.num_count = 0
        self.count = 0
        
        
    def cb_digit(self,msg):
        self.digit = msg.data
        
    def callback_status(self,msg):
        self.status = msg.data
        
    def rcv_img(self, msg):
        image = self.jpeg.decode(msg.data)
        self.img_queue.append(image)
    
    def run(self): 
        rate = rospy.Rate(5) # 5Hz
        while not rospy.is_shutdown():
            if len(self.digit_detections) == 10:
                self.id_pub.publish(9999)
                rospy.signal_shutdown("Done")
            if self.img_queue:
                img = self.img_queue.popleft()  
                self.process_img(img)
            rate.sleep()
    
    def draw_detect_results(self, img, tag, dist):
        # Enumerate through the detection results
        (ptA, ptB, ptC, ptD) = tag.corners

        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        
        # cropped_img
        tag_width = ptB[0] - ptA[0]
        #1.5
        top_dist = int(tag_width * 1.5)
        x1 = int(ptD[1]-top_dist)
        if x1 < 0:
            x1 = 0
            
        x2 = ptD[1]
        if x2 >= 480:
            x2 = 479
        
        x3 = ptA[0]-25
        if x3 < 0:
            x3 = 0
            
        x4 = ptB[0]+25
        if x4 >= 640:
            x4 = 639
        
        cropped_img = img[x1:x2, x3:x4]
        h, w = cropped_img.shape
        # rospy.loginfo(f'width: {w} height: {h}')
        if h == 0:
            return None
        
        # # collect images for training
        # filename = '/data/bags/n'+ str(self.counter)+".jpg"
        # cv2.imwrite(filename, cropped_img)
        # self.counter = self.counter + 1
        
        # A = (x3, x2)
        # B = (x4, x2)
        # C = (x4, x1)
        # D = (x3, x1)

        # # Determine the decoded ID of tag, and decide the color of bbox 
        # id = tag.tag_id

        # tag_name = "Stop sign"
        # which_tag = 's'
        # col = (0, 0, 255)   # stop sign (default)
        # if id in [153, 58, 133, 62]:
        #     tag_name = "T-intersection"
        #     col = (255, 0, 0)   # T-intersection sign
        # elif id in [201, 93, 94, 200]:
        #     tag_name = "UofA"name
        # cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.putText(img, tag_name, (cx, cy + 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)

        return cropped_img
    
    
    def process_img(self, img):
        target_size = self.detect_intersection(img)
        if target_size > 0.2 and self.next_intersection != -1:
            # rospy.loginfo(f'At intersection, target size: {target_size}')
            delen = len(self.detections)
            # rospy.loginfo(f'Number of detections: {delen}')
            self.id_pub.publish(self.next_intersection)
            self.next_intersection = -1
            # Todo: handle intersection
        else:
            undistorted_img = self.undistort_img(img)
            tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size)                

            min_dist = -1
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
                # rospy.loginfo(f'Tag ID: {min_tag_id}')
                # rospy.loginfo(f'dist: {min_dist}')
                
                if self.count > 0:
                    if undistorted_img is not None:
                        self.count += 1
                        if self.count == 3:
                            image = self.bridge.cv2_to_compressed_imgmsg(undistorted_img)
                            drawed_image = self.draw_detect_results(undistorted_img, min_tag, min_dist)
                            if drawed_image is not None:
                                drawed_image = self.bridge.cv2_to_compressed_imgmsg(drawed_image)
                                # self.pub_view.publish(drawed_image)
                                self.image_pub.publish(drawed_image)   # send image data to digit detector
                                time.sleep(7)
                                
                                # self.digit_detections.append(7)
                                if self.digit != -1 and self.digit not in self.digit_detections:
                                    self.digit_detections.append(self.digit)
                        
                                rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                                rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                                rospy.loginfo(f'Tag ID: {min_tag_id}')
                                rospy.loginfo(f'Tag location: {self.tag_info[min_tag_id]}')
                                rospy.loginfo(f'Detected digit: {self.digit}')
                                rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                                rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                                
                            self.count = 0
                
                if min_tag_id not in self.detections and  min_dist < self.dist_threshold:
                    self.count = 1
                    if min_tag_id == 94:
                        self.id_pub.publish(min_tag_id)
                    else:
                        self.id_pub.publish(STOP)
                        
                    self.detections.append(min_tag_id)
                
                if self.next_intersection == -1 and min_tag_id in self.turning_tags:
                        self.next_intersection = min_tag_id
                        # rospy.loginfo(f'Intersection set to: {min_tag_id}')
                        

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
