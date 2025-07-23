import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import message_filters
import struct


class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_converter')
        
        # 声明参数以方便从launch文件或命令行配置话题名称
        self.declare_parameter('depth_topic', '/camera/depth_registered/image_rect')
        self.declare_parameter('info_topic', '/camera/depth_registered/camera_info')   #用这两个
        self.declare_parameter('output_topic', '/depth_camera/points')
        
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # 创建点云发布者
        self.pc_publisher = self.create_publisher(PointCloud2, output_topic, 10)
        
        # 创建订阅者
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)
        
        # 使用TimeSynchronizer同步消息
        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.info_sub], 10)
        self.ts.registerCallback(self.image_callback)
        
        self.bridge = CvBridge()
        self.camera_model_ready = False
        self.get_logger().info('Depth to PointCloud converter node started.')

    def image_callback(self, depth_msg, info_msg):
        # 检查相机模型是否已初始化 (只需要一次)
        if not self.camera_model_ready:
            self.K = np.array(info_msg.k).reshape(3, 3)
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.camera_model_ready = True
            self.get_logger().info(f"Camera model initialized with fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

        try:
            # 将深度图像从ROS消息转换为OpenCV/Numpy格式
            # 假设深度单位是米 (32FC1) 或 毫米 (16UC1)
            if depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            elif depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                depth_image = depth_image.astype(np.float32) / 1000.0 # 转换毫米到米
            else:
                self.get_logger().error(f"Unsupported depth format: {depth_msg.encoding}")
                return
        except Exception as e:
            self.get_logger().error(f"Could not convert depth image: {e}")
            return
            
        height, width = depth_image.shape
        points = []

        # 这是一个简单但低效的循环方法，用于教学目的
        # 在实际应用中，推荐使用Numpy的向量化操作以提高性能
        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                # 忽略无效的深度值 (通常为0或NaN)
                if z > 0:
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    points.append([x, y, z])
        
        # 创建PointCloud2消息
        header = depth_msg.header # 使用深度图的header
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # 将点列表打包成二进制数据
        # 注意: ROS 2中没有像ROS 1中那么方便的pc2.create_cloud_xyz32工具
        # 我们需要手动打包
        point_struct = struct.Struct('<fff')
        packed_points = [point_struct.pack(*p) for p in points]
        
        pc2_msg = PointCloud2(
            header=header,
            height=1, # 对于无序点云，高度为1
            width=len(points),
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=12, # 3 * 4 bytes (float32)
            row_step=12 * len(points),
            data=b"".join(packed_points)
        )
        
        self.pc_publisher.publish(pc2_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()