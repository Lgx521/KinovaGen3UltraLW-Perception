#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
import threading
from scipy.spatial.transform import Rotation as R

# --- 用户可配置参数 ---
# ArUco码的物理边长（单位：米）。请务必修改为您的ArUco码的实际尺寸！
MARKER_SIZE = 0.035  # 示例: 5厘米
# ArUco字典类型
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# ROS话题名称
IMAGE_TOPIC = '/camera/color/image_raw'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
# ---------------------

# 使用一个Event来通知主线程处理已完成
processing_done_event = threading.Event()

def synchronized_callback(image_msg, camera_info_msg):
    """
    当收到同步的图像和相机信息时，此回调函数被调用。
    处理完第一组有效数据后，将设置一个事件以终止程序。
    """
    global processing_done_event
    
    print("Received synchronized image and camera info.")
    bridge = CvBridge()

    # 1. 从CameraInfo消息中提取相机内参和畸变系数
    camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
    dist_coeffs = np.array(camera_info_msg.d)

    # 2. 将ROS图像消息转换为OpenCV图像
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(f"CvBridge Error: {e}")
        processing_done_event.set() # 出错也终止程序
        return

    # 3. 使用cv2检测ArUco码
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        cv_image, ARUCO_DICT, parameters=aruco_params)

    # 4. 如果检测到ArUco码，则计算位姿并打印
    if ids is not None and len(ids) > 0:
        print(f"Found {len(ids)} markers.")
        # 估计每个标记的位姿 (rvecs: 旋转向量, tvecs: 平移向量)
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, camera_matrix, dist_coeffs)

        # 遍历所有检测到的ArUco码
        for i, marker_id in enumerate(ids):
            # 平移向量 (tvec) 就是相机坐标系中的位置
            tvec = tvecs[i][0]
            
            # 旋转向量 (rvec) 需要转换为四元数
            rvec = rvecs[i][0]
            rotation = R.from_rotvec(rvec)
            quat = rotation.as_quat() # 返回 [x, y, z, w] 格式的四元数

            print(f"\n--- Marker ID: {marker_id[0]} ---")
            # 按照指定格式打印结果
            print(f"  position: {{ x: {tvec[0]:.4f}, y: {tvec[1]:.4f}, z: {tvec[2]:.4f} }},")
            print(f"  orientation: {{ x: {quat[0]:.4f}, y: {quat[1]:.4f}, z: {quat[2]:.4f}, w: {quat[3]:.4f} }}")
            print(f"  orientation_vec: {{ rx: {rvec[0]:.4f}, ry: {rvec[1]:.4f}, rz: {rvec[2]:.4f} }},")
    else:
        print("No ArUco markers found in the current frame.")

    # 通知主线程可以退出了
    processing_done_event.set()


def main(args=None):
    rclpy.init(args=args)
    # ========================== 修改之处 ==========================
    # 移除了 'anonymous=True' 参数以兼容较旧的 rclpy 版本
    node = rclpy.create_node('aruco_pose_getter')
    # ============================================================
    
    print("Aruco Pose Getter Script")
    print(f"Listening for topics:\n  - Image: {IMAGE_TOPIC}\n  - Camera Info: {CAMERA_INFO_TOPIC}")
    print(f"Using marker size: {MARKER_SIZE} meters")
    print("Waiting for a synchronized message pair...")

    # 设置订阅器
    image_sub = message_filters.Subscriber(node, Image, IMAGE_TOPIC)
    cam_info_sub = message_filters.Subscriber(node, CameraInfo, CAMERA_INFO_TOPIC)

    # 同步订阅的话题
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, cam_info_sub], 10, 0.1)
    ts.registerCallback(synchronized_callback)

    # 循环等待，直到回调函数发出完成信号
    while rclpy.ok() and not processing_done_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)

    print("\nProcessing finished. Shutting down.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()