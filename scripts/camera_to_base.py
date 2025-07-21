#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class PoseTransformer(Node):
    def __init__(self):
        super().__init__('pose_transformer_node')
        
        # --- 1. 定义需要变换的位姿 ---
        # 
        # ========================== 坐标系定义说明 ==========================
        # 此处定义的位姿，是基于您指定的相机坐标系标准。
        # 这个标准被称为“相机光学坐标系”(Camera Optical Frame)，也是OpenCV的标准。
        #
        #   - +X 轴: 指向图像的右方
        #   - +Y 轴: 指向图像的下方
        #   - +Z 轴: 从相机镜头直直向前，指向场景深处
        #
        # ====================================================================
        
        self.source_frame = 'camera_link' # 源坐标系名称
        self.target_frame = 'base_link'   # 目标坐标系名称

        # 示例位姿：定义一个点在 camera_link 坐标系下的位置和方向
        # 位置 (x, y, z) 单位: 米
        # 示例: 在相机右侧0.1m, 下方0.05m, 前方0.5m处的一个点
        self.source_position = np.array([-0.00441, 0.004495, 0.293]) 
        
        # 方向 (四元数: x, y, z, w)
        # 示例: 无旋转 (与相机坐标系方向一致)
        self.source_orientation_quat = np.array([0.29491, 0.293475, 0.6445674, -0.641430664])

        self.get_logger().info(f"Transforming a pose from '{self.source_frame}' to '{self.target_frame}'.")
        self.get_logger().info(
            f"Source Pose in '{self.source_frame}' (using Camera Optical convention):\n"
            f"  position: {{ x: {self.source_position[0]}, y: {self.source_position[1]}, z: {self.source_position[2]} }},\n"
            f"  orientation: {{ x: {self.source_orientation_quat[0]}, y: {self.source_orientation_quat[1]}, z: {self.source_orientation_quat[2]}, w: {self.source_orientation_quat[3]} }}"
        )

        # --- 2. 初始化 TF 监听器 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def transform_pose(self):
        """查询TF并执行坐标变换"""
        
        # --- 3. 等待并查询 TF 变换 ---
        self.get_logger().info("Waiting for the transform to become available...")
        while rclpy.ok():
            try:
                # 获取从源坐标系(camera_link)到目标坐标系(base_link)的变换
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                self.get_logger().info("Transform found!")
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"Could not get transform: {e}. Retrying...", throttle_duration_sec=1.0)
                time.sleep(1)

        # --- 4. 执行坐标变换 ---
        # 变换的数学逻辑是通用的，不依赖于坐标轴的具体指向，只依赖于TF的正确性
        tf_translation = transform.transform.translation
        tf_rotation = transform.transform.rotation
        
        pos_target_source = np.array([tf_translation.x, tf_translation.y, tf_translation.z])
        rot_target_source = R.from_quat([tf_rotation.x, tf_rotation.y, tf_rotation.z, tf_rotation.w])

        rot_source_point = R.from_quat(self.source_orientation_quat)

        # 计算变换后的位姿
        rot_final = rot_target_source * rot_source_point
        pos_final = pos_target_source + rot_target_source.apply(self.source_position)

        final_position = pos_final
        final_quat = rot_final.as_quat()

        # --- 5. 打印结果 ---
        self.get_logger().info(
            f"\n--- Transformed Pose in '{self.target_frame}' ---\n"
            f"  position: {{ x: {final_position[0]:.4f}, y: {final_position[1]:.4f}, z: {final_position[2]:.4f} }},\n"
            f"  orientation: {{ x: {final_quat[0]:.4f}, y: {final_quat[1]:.4f}, z: {final_quat[2]:.4f}, w: {final_quat[3]:.4f} }}\n"
        )
        return True

def main(args=None):
    rclpy.init(args=args)
    pose_transformer_node = PoseTransformer()
    pose_transformer_node.transform_pose()
    pose_transformer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()