import rclpy
from rclpy.node import Node
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import struct

class ColorPointCloudGenerator(Node):
    """
    一个将已配准的深度图和彩色图合成为彩色点云的ROS 2节点。
    """
    def __init__(self):
        super().__init__('color_pointcloud_generator')

        # --- 参数定义 ---
        # 使用declare_parameter来让话题名称等配置更加灵活
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_topic', '/camera/points_colored')
        
        # 获取参数值
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # --- 成员变量初始化 ---
        self.bridge = CvBridge()
        self.camera_intrinsics = None # 用于存储相机内参

        # --- 发布者 ---
        self.pc_publisher = self.create_publisher(PointCloud2, output_topic, 10)

        # --- 订阅者与消息同步 ---
        # 使用message_filters来同步彩色图、深度图和相机信息
        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)

        # 使用ApproximateTimeSynchronizer，因为它对时间戳的微小差异更具鲁棒性
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=10, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.pointcloud_callback)

        self.get_logger().info('彩色点云生成节点已启动...')
        self.get_logger().info(f'订阅彩色图像: {color_topic}')
        self.get_logger().info(f'订阅深度图像: {depth_topic}')
        self.get_logger().info(f'订阅相机信息: {camera_info_topic}')
        self.get_logger().info(f'发布点云话题: {output_topic}')


    def pointcloud_callback(self, color_msg, depth_msg, info_msg):
        """
        同步消息的回调函数，处理图像并生成点云。
        """
        # --- 1. 获取相机内参 (仅在第一次时处理) ---
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': info_msg.k[0],
                'fy': info_msg.k[4],
                'cx': info_msg.k[2],
                'cy': info_msg.k[5],
                'width': info_msg.width,
                'height': info_msg.height
            }
            self.get_logger().info(f"相机内参已获取: {self.camera_intrinsics}")

        # --- 2. 将ROS Image消息转换为OpenCV/Numpy格式 ---
        try:
            # 彩色图通常是 'bgr8'
            color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            # 深度图通常是 '16UC1' (毫米) 或 '32FC1' (米)
            if depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
                # 将毫米转换为米
                depth_image = depth_image.astype(np.float32) / 1000.0
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            else:
                self.get_logger().error(f"不支持的深度图像编码: {depth_msg.encoding}")
                return
        except Exception as e:
            self.get_logger().error(f"转换图像失败: {e}")
            return

        # --- 3. 核心计算：从像素生成3D点 ---
        # 使用Numpy向量化操作，避免Python的for循环，性能更高
        height, width = depth_image.shape
        fx, fy, cx, cy = (self.camera_intrinsics['fx'], self.camera_intrinsics['fy'], 
                          self.camera_intrinsics['cx'], self.camera_intrinsics['cy'])

        # 创建像素坐标网格
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))

        # 获取深度值 Z
        Z = depth_image

        # 过滤掉无效的深度值 (0 或 NaN)
        valid_mask = (Z > 0) & np.isfinite(Z)

        # 反投影公式计算 X 和 Y
        X = np.where(valid_mask, (u_grid - cx) * Z / fx, 0)
        Y = np.where(valid_mask, (v_grid - cy) * Z / fy, 0)
        
        # 将X, Y, Z和颜色信息组合起来
        points_3d = np.dstack((X, Y, Z))[valid_mask]
        colors_bgr = color_image[valid_mask]

        # --- 4. 构建 PointCloud2 消息 ---
        # 定义点云字段。我们需要x, y, z和rgb
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # 将BGR颜色打包成一个UINT32整数
        # OpenCV的BGR格式是(Blue, Green, Red)
        # 我们需要打包成 0x00RRGGBB 的格式
        B = colors_bgr[:, 0].astype(np.uint32)
        G = colors_bgr[:, 1].astype(np.uint32)
        R = colors_bgr[:, 2].astype(np.uint32)
        rgb_packed = (R << 16) | (G << 8) | B

        # 将点数据和颜色数据打包成二进制
        points_data = []
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            rgb = rgb_packed[i]
            # struct.pack将数据打包成字节流, '<'表示小端序
            points_data.append(struct.pack('<fffI', x, y, z, rgb))
        
        # 创建PointCloud2消息头
        header = color_msg.header # 关键！使用彩色图的header，包含正确的frame_id和timestamp

        # 创建并填充PointCloud2消息
        pc_msg = PointCloud2(
            header=header,
            height=1,
            width=len(points_3d),
            is_dense=False, # 因为我们过滤掉了一些点
            is_bigendian=False,
            fields=fields,
            point_step=16, # 4字节*3(xyz) + 4字节(rgb) = 16字节
            row_step=16 * len(points_3d),
            data=b"".join(points_data)
        )

        # --- 5. 发布点云 ---
        self.pc_publisher.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ColorPointCloudGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()