#!/usr/bin/env python3

import os
import numpy as np
import cv2
import yaml
import tf
import math
import numpy as np
from cv_bridge import CvBridge
from collections import deque
from turbojpeg import TurboJPEG, TJPF_GRAY
from dt_apriltags import Detector
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from duckietown_msgs.srv import GetVariable

import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Transform, Vector3, Quaternion
from std_msgs.msg import String, Int32


class TestNode(DTROS):
    def __init__(self, node_name):
        super(TestNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        name = os.environ['VEHICLE_NAME']
        self.image_sub = rospy.Subscriber(f'/{name}/camera_node/image/compressed', CompressedImage, self.rcv_img,  queue_size = 1)
        self.image_pub = rospy.Publisher(f'/{name}/digit_detector_node/image/compressed', CompressedImage,  queue_size = 1)
        self.digit_sub = rospy.Subscriber(f'/{name}/digit', Int32, self.cb_digit,  queue_size = 1)
        self.img_queue = deque(maxlen=1)
        self.rectify_alpha = rospy.get_param("~rectify_alpha", 0.0)

        self.bridge = CvBridge() 
        self.jpeg = TurboJPEG()
        
        self._map_xy_set = False
        
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

        self.detector =  Detector(
            families=self.family,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,) 
        

        self.dist_threshold = 10
        self.counter = 0
        self.digit = -1
    
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
        

    def cb_digit(self,msg):
        self.digit = msg.data
        
    def rcv_img(self, msg):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.img_queue.append(image)
        # rospy.loginfo('Image received...')

    def run(self): 
        rate = rospy.Rate(15) # 5Hz
        while not rospy.is_shutdown():
            if self.img_queue:
                img = self.img_queue.popleft()  
                # self.detect_duck(img)
                self.detect_bot(img)
            rate.sleep()
                
    
    def detect_bot(self, img):
        h, w, d = img.shape
        # crop = img[300:-1, 0:int(0.65*w), :]
        crop = img
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40,50,0])
        upper_green = np.array([130,255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        target_size = np.sum(mask/255.) / mask.size
        rospy.loginfo(f'bot mask:{target_size}')
        return target_size
        # if target_size > 0.1:
        #     return True
        # else:
        #     return False
        
    def detect_ducks(self, img):
        # Enumerate through the detection results
        h, w, d = img.shape
        crop = img[360:-1, 0:int(0.65*w), :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # lower_blue = np.array([80,50,0])
        # upper_blue = np.array([130,255, 255])
        # mask0 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_orange = np.array([0,0,0])
        upper_orange = np.array([50,255, 255])
        mask1 = cv2.inRange(hsv, lower_orange, upper_orange)

        # mask = mask0 + mask1
        target_size = np.sum(mask1/255.) / mask1.size
        
        
        
        # mask_or = cv2.bitwise_or(mask0, mask1)
        # target_size1 = np.sum(mask_or/255.) / mask_or.size
        
        # mask_and = cv2.bitwise_and(mask0, mask1)
        # target_size2 = np.sum(mask_and/255.) / mask_and.size
        # target = cv2.bitwise_and(crop, crop, mask = mask)
        
        # rospy.loginfo(f'1: {target_size1} 2:{target_size2}')
        
        # if DEBUG:
        #     rect_img_msg = self.bridge.cv2_to_compressed_imgmsg(cropped_img)
        #     self.image_pub.publish(rect_img_msg)
            
            
        # rospy.loginfo(f'width: {w} height: {h}')
        # if h == 0:
        #     return None
        
        return target_size
    
    def detect_duck(self, img):
       
        # undistorted_img = self.undistort_img(img)
        detect = self.detect_ducks(img)
        rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
        rospy.loginfo(f'Ducks detected: {detect}')
        rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')


    def handle_state(req):
        return 

    def image_server():
        rospy.init_node('image_server')
        s = rospy.Service('image_srv', GetVariable, handle_state)
        rospy.spin()
    

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
    test_node = TestNode(node_name='test_node')
    test_node.run()
    rospy.spin()
