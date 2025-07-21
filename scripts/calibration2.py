#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
from scipy.spatial.transform import Rotation as R

class AprilTagDetector(Node):
    def __init__(self):
        # 移除了 anonymous=True 以兼容旧版ROS
        super().__init__('apriltag_detector_node')
        self.get_logger().info("AprilTag Live Detector Node has been started.")

        # --- 1. 参数和配置 ---
        # 声明并获取参数
        self.declare_parameter('marker_size', 0.05)  # AprilTag的物理边长（米）, 默认5cm
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('info_topic', '/camera/color/camera_info')
        
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value

        self.get_logger().info(f"Using AprilTag size: {self.marker_size} meters")
        self.get_logger().info(f"Subscribing to image topic: {image_topic}")
        self.get_logger().info(f"Subscribing to camera info topic: {info_topic}")

        # 设置AprilTag字典为 TAG_25h9
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # --- 2. 初始化工具和状态变量 ---
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False

        # --- 3. 设置同步订阅 ---
        self.image_sub = message_filters.Subscriber(self, Image, image_topic)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cam_info_sub], 10, 0.1)
        self.ts.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, image_msg, camera_info_msg):
        """
        同步回调函数，处理图像和相机信息。
        """
        # 仅在第一次接收到消息时存储相机参数
        if not self.camera_info_received:
            self.camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info_msg.d)
            self.camera_info_received = True
            self.get_logger().info("Camera info received and stored.")

        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # 检测AprilTag
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dict, parameters=self.aruco_params)

        # 如果检测到，则进行处理
        if ids is not None and len(ids) > 0:
            # 在图像上绘制检测到的标记边界
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            # 估计位姿
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            # 遍历每个检测到的标记
            for i, marker_id in enumerate(ids):
                # 在图像上绘制3D坐标轴，以可视化方向
                # 注意: 老版本OpenCV使用 cv2.aruco.drawAxis
                try:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size * 0.75)
                except AttributeError:
                    cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_size * 0.75)

                # 打印位姿信息到终端
                self.print_pose(tvecs[i], rvecs[i], marker_id)
        
        # 显示带有结果的实时图像
        cv2.imshow("AprilTag Detector", cv_image)
        # 必须有waitKey，否则窗口不刷新
        cv2.waitKey(1)

    def print_pose(self, tvec, rvec, marker_id):
        """
        格式化并打印位姿信息。
        """
        # 平移向量 tvec 就是位置
        position = tvec.ravel()
        
        # 旋转向量 rvec 转换为四元数
        rotation = R.from_rotvec(rvec.ravel())
        quat = rotation.as_quat() # [x, y, z, w]

        self.get_logger().info(
            f"--- Marker ID: {marker_id[0]} ---\n"
            f"  position: {{ x: {position[0]:.4f}, y: {position[1]:.4f}, z: {position[2]:.4f} }},\n"
            f"  orientation: {{ x: {quat[0]:.4f}, y: {quat[1]:.4f}, z: {quat[2]:.4f}, w: {quat[3]:.4f} }}"
        )

def main(args=None):
    rclpy.init(args=args)
    detector_node = AprilTagDetector()
    
    try:
        # 使用spin会让节点持续运行，接收和处理消息
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info("Keyboard Interrupt (Ctrl+C). Shutting down.")
    finally:
        # 清理
        detector_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()