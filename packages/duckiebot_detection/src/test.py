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
        
        # # Publisher
        # self.pub_view = rospy.Publisher(
        #     f'/{name}/test/image/compressed',
        #     CompressedImage,
        #     queue_size=1
        # )
        
        self.counter = 1
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
                self.detect_tag(img)
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
        top_dist = int(tag_width * 1.5)
        x1 = int(ptD[1]-top_dist)
        if x1 < 0:
            x1 = 0
            
        x2 = ptD[1]
        if x2 >= 480:
            x2 = 479
        
        x3 = ptA[0] - 10
        if x3 < 0:
            x3 = 0
            
        x4 = ptB[0] + 10
        if x4 >= 640:
            x4 = 639
        
        cropped_img = img[x1:x2, x3:x4]
        h, w = cropped_img.shape
        # rospy.loginfo(f'width: {w} height: {h}')
        if h == 0:
            return None
        
    
        # collect images for training
        filename = '/data/bags/nn'+ str(self.counter)+".jpg"
        cv2.imwrite(filename, cropped_img)
        self.counter = self.counter + 1

        # A = (x3, x2)
        # B = (x4, x2)
        # C = (x4, x1)
        # D = (x3, x1)
        
        # # Draw the bbox
        # col = (0, 0, 255)
        # cv2.line(cropped_img, A, B, col, 2)
        # cv2.line(cropped_img, B, C, col, 2)
        # cv2.line(cropped_img, C, D, col, 2)
        # cv2.line(cropped_img, D, A, col, 2)
        
        
        # Determine the decoded ID of tag, and decide the color of bbox 
        # id = tag.tag_id

        # tag_name = "Stop sign"
        # which_tag = 's'
        # col = (0, 0, 255)   # stop sign (default)
        # if id in [153, 58, 133, 62]:
        #     tag_name = "T-intersection"
        #     col = (255, 0, 0)   # T-intersection sign
        # elif id in [201, 93, 94, 200]:
        #     tag_name = "UofA"
        #     col = (0, 255, 0)   # UofA sign
        
        # # Draw the center dot and name of tag
        # cx, cy = int(tag.center[0]), int(tag.center[1])
        # cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        # cv2.putText(img, tag_name, (cx, cy + 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)

        return cropped_img
    
    
    def detect_tag(self, img):
        undistorted_img = self.undistort_img(img)
        tags = self.detector.detect(undistorted_img, True, self._camera_parameters, self.tag_size)
        
        # img_remap = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)
        # rospy.loginfo(f'shape: {img_remap.shape}')

        min_dist = -1
        min_tag_id = -1 
        min_tag = None
        
        for tag in tags:
            p = tag.pose_t.T[0]
            tag_id=int(tag.tag_id)
            dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

            if dist > self.dist_threshold:
                continue

            if min_dist == -1:
                min_dist = dist
                min_tag_id = tag_id 
                min_tag = tag
            elif dist < min_dist:
                min_dist = dist
                min_tag_id = tag_id 
                min_tag = tag

        if min_tag_id != -1: # We have the tag with minimum distance
            # gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            
            # rospy.loginfo(f'Tag : {min_tag}')
            # rospy.loginfo(f'dist: {min_dist}')
            
            if undistorted_img is not None:
                image = self.bridge.cv2_to_compressed_imgmsg(undistorted_img)
                drawed_image = self.draw_detect_results(undistorted_img, min_tag, min_dist)
                if drawed_image is not None:
                    drawed_image = self.bridge.cv2_to_compressed_imgmsg(drawed_image)
                    # self.pub_view.publish(drawed_image)
                    self.image_pub.publish(drawed_image)   # send image data to digit detector
                    
                    rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                    rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
                    rospy.loginfo(f'Tag ID: {min_tag_id}')
                    rospy.loginfo(f'Tag location: {self.tag_info[min_tag_id]}')
                    rospy.loginfo(f'Detected digit: {self.digit}')
                    rospy.loginfo(f'_*_*_*_*_*_*_*_*_*_*_*_*_')
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
