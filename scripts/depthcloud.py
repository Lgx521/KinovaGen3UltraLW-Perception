# color_pointcloud_generator_optimized.py

import rclpy
from rclpy.node import Node
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge

# 这是关键，我们定义一个结构化的numpy数据类型来匹配PointCloud2的格式
# 这样可以一次性将所有数据转换为字节流，而不是用for循环和struct.pack
from ros2_numpy.point_cloud2 import array_to_pointcloud2

class ColorPointCloudGeneratorOptimized(Node):
    def __init__(self):
        super().__init__('color_pointcloud_generator_optimized')

        # 参数定义和之前一样
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_topic', '/camera/points_colored')
        
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.camera_intrinsics = None
        
        self.pc_publisher = self.create_publisher(PointCloud2, output_topic, 10)

        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)
        
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10, slop=0.2 # slop可以适当调大一点
        )
        self.time_synchronizer.registerCallback(self.pointcloud_callback)

        self.get_logger().info('优化的彩色点云生成节点已启动...')

    def pointcloud_callback(self, color_msg, depth_msg, info_msg):
        # 1. 获取相机内参 (仅一次)
        if self.camera_intrinsics is None:
            self.K = np.array(info_msg.k).reshape(3, 3)
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.camera_intrinsics = True
            self.get_logger().info(f"相机内参已获取: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

        # 2. 图像转换
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            if depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            else:
                self.get_logger().error(f"不支持的深度图像编码: {depth_msg.encoding}")
                return
        except Exception as e:
            self.get_logger().error(f"转换图像失败: {e}")
            return
        
        # --- 核心优化部分 ---
        height, width = depth_image.shape
        
        # 过滤掉无效深度值
        valid_mask = (depth_image > 0) & np.isfinite(depth_image)

        # 1. 创建像素坐标网格 (Vectorized)
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))

        # 2. 反投影计算 (Vectorized)
        Z = depth_image
        X = np.where(valid_mask, (u_grid - self.cx) * Z / self.fx, 0)
        Y = np.where(valid_mask, (v_grid - self.cy) * Z / self.fy, 0)
        
        # 3. 将XYZ坐标和颜色组合成结构化数组 (Vectorized)
        xyz = np.dstack((X, Y, Z))[valid_mask]
        rgb = color_image[valid_mask]
        
        # 创建一个结构化的numpy数组
        # 'names'定义了字段名, 'formats'定义了每个字段的数据类型
        # 'f4'是4字节浮点数 (float32), 'u1'是1字节无符号整数 (uint8)
        points_struct = np.zeros(xyz.shape[0], dtype={
            'names': ('x', 'y', 'z', 'b', 'g', 'r', 'a'),
            'formats': ('f4', 'f4', 'f4', 'u1', 'u1', 'u1', 'u1'),
            'offsets': (0, 4, 8, 12, 13, 14, 15),
            'itemsize': 16
        })
        
        points_struct['x'] = xyz[:, 0]
        points_struct['y'] = xyz[:, 1]
        points_struct['z'] = xyz[:, 2]
        points_struct['b'] = rgb[:, 0]
        points_struct['g'] = rgb[:, 1]
        points_struct['r'] = rgb[:, 2]
        points_struct['a'] = 255 # Alpha通道

        # --- 使用ros2_numpy将结构化数组高效转换为PointCloud2消息 ---
        # 确保你已经安装了: pip install ros2-numpy
        pc_msg = array_to_pointcloud2(points_struct, color_msg.header)

        self.pc_publisher.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    # 安装ros2_numpy: pip install ros2-numpy
    try:
        from ros2_numpy.point_cloud2 import array_to_pointcloud2
    except ImportError:
        print("请安装 ros2_numpy: pip install ros2-numpy")
        return

    node = ColorPointCloudGeneratorOptimized()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()