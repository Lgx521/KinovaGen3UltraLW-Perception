#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R

# 引入 TF2 相关库
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector_node')
        self.get_logger().info("AprilTag Live Detector with TF Transformation has been started.")

        # --- 1. 参数和配置 ---
        self.declare_parameter('marker_size', 0.035)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('info_topic', '/camera/color/camera_info')
        # 新增TF相关参数
        self.declare_parameter('camera_frame', 'camera_link') # 相机TF坐标系名称
        self.declare_parameter('target_frame', 'base_link') # 目标TF坐标系名称

        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.get_logger().info(f"AprilTag size: {self.marker_size}m")
        self.get_logger().info(f"Target frame: '{self.target_frame}', Camera frame: '{self.camera_frame}'")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # --- 2. 初始化工具和状态变量 ---
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # --- 3. 初始化 TF 监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 4. 设置同步订阅 ---
        self.image_sub = message_filters.Subscriber(self, Image, image_topic)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cam_info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, image_msg, camera_info_msg):
        # 仅在第一次接收到消息时存储相机参数
        if not self.camera_info_received:
            self.camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info_msg.d)
            self.camera_info_received = True
            self.get_logger().info("Camera info received and stored.")

        # --- 查询TF变换 ---
        try:
            # 获取从相机坐标系到目标坐标系(base_link)的变换
            # 使用 image_msg.header.stamp 可以获取更精确的同步变换，但最新变换通常也足够
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time()) # 获取最新的可用变换
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Could not get transform from {self.camera_frame} to {self.target_frame}: {e}')
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        corners, ids, _ = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i, marker_id in enumerate(ids):
                # 可视化绘制
                try:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size * 0.75)
                except AttributeError:
                    cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size * 0.75)

                # --- 进行坐标变换 ---
                self.transform_and_print_pose(transform, tvecs[i], rvecs[i], marker_id)
        
        cv2.imshow("AprilTag Detector", cv_image)
        cv2.waitKey(1)

    def transform_and_print_pose(self, transform: TransformStamped, tvec, rvec, marker_id):
        """
        将位姿从相机坐标系变换到目标坐标系并打印。
        """
        # 1. 将相机到Tag的位姿转换为Scipy的Rotation对象
        pos_cam_tag = tvec.ravel()
        rot_cam_tag = R.from_rotvec(rvec.ravel())

        # 2. 将base_link到相机的TF变换也转换为Scipy的Rotation对象
        tf_translation = transform.transform.translation
        tf_rotation = transform.transform.rotation
        pos_base_cam = np.array([tf_translation.x, tf_translation.y, tf_translation.z])
        rot_base_cam = R.from_quat([tf_rotation.x, tf_rotation.y, tf_rotation.z, tf_rotation.w])

        # 3. 计算Tag在base_link中的位姿
        # 旋转: R_base_tag = R_base_cam * R_cam_tag
        rot_base_tag = rot_base_cam * rot_cam_tag
        # 平移: p_base_tag = p_base_cam + R_base_cam * p_cam_tag
        pos_base_tag = pos_base_cam + rot_base_cam.apply(pos_cam_tag)

        # 4. 获取最终结果
        final_position = pos_base_tag
        final_quat = rot_base_tag.as_quat() # [x, y, z, w]

        rot_base_tag = rot_base_tag.as_rotvec()

        # 5. 按指定格式打印
        self.get_logger().info(
            f"--- Pose of Marker ID: {marker_id[0]} in '{self.target_frame}' frame ---\n"
            f"  position: {{ x: {final_position[0]:.4f}, y: {final_position[1]:.4f}, z: {final_position[2]:.4f} }},\n"
            f"  orientation: {{ x: {final_quat[0]:.6f}, y: {final_quat[1]:.6f}, z: {final_quat[2]:.6f}, w: {final_quat[3]:.6f} }}\n"
            f"  orientation_ang: {{ rx: {rot_base_tag[0]:.4f}, ry: {rot_base_tag[1]:.4f}, rz: {rot_base_tag[2]:.4f} }}"
        )

def main(args=None):
    rclpy.init(args=args)
    detector_node = AprilTagDetector()
    
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info("Keyboard Interrupt (Ctrl+C). Shutting down.")
    finally:
        detector_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()