#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

# 使用我们之前讨论过的低延迟多线程视频读取类
class VideoStream:
    """A class to read frames from a camera in a separate thread."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Cannot open video stream at {src}")
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

class RtspStreamPublisher(Node):
    """
    This node captures a video stream from an RTSP source using a low-latency
    threaded reader and publishes it as a sensor_msgs/Image topic.
    """
    def __init__(self):
        super().__init__('rtsp_stream_publisher')
        
        # --- 参数声明 ---
        self.declare_parameter('rtsp_url', 'rtsp://192.168.1.10/color')
        self.declare_parameter('topic_name', '/kinova_camera/color/image_raw')
        self.declare_parameter('publish_rate', 30.0)

        # 获取参数
        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.get_logger().info(f"Connecting to RTSP URL: {rtsp_url}")
        self.get_logger().info(f"Publishing to topic: {topic_name}")
        self.get_logger().info(f"Publish rate: {publish_rate} Hz")

        # --- 初始化 ---
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        
        # 启动视频流读取线程
        try:
            self.video_stream = VideoStream(rtsp_url)
            time.sleep(2.0) # 给视频流一些启动时间
        except IOError as e:
            self.get_logger().error(f'Failed to start video stream: {e}')
            rclpy.shutdown()
            return
            
        # 创建一个定时器，以指定的频率调用 publisher_callback
        self.timer = self.create_timer(1.0 / publish_rate, self.publisher_callback)
        self.get_logger().info('RTSP stream publisher node has been started.')

    def publisher_callback(self):
        """
        This function is called periodically. It reads the latest frame from the
        video stream, converts it to a ROS Image message, and publishes it.
        """
        # 从后台线程获取最新的一帧
        frame = self.video_stream.read()

        if frame is not None:
            # 将OpenCV的图像 (BGR) 转换为ROS Image消息
            # CvBridge 会自动处理编码和时间戳
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            
            # 使用当前ROS时间作为消息头的时间戳
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            ros_image_msg.header.frame_id = "kinova_camera_link" # 可以自定义frame_id

            # 发布消息
            self.publisher_.publish(ros_image_msg)
        else:
            self.get_logger().warn('Could not read frame from video stream.')

    def on_shutdown(self):
        """
        Cleanup function called on node shutdown.
        """
        self.get_logger().info('Shutting down, stopping video stream...')
        self.video_stream.stop()

def main(args=None):
    rclpy.init(args=args)
    rtsp_stream_publisher = RtspStreamPublisher()
    
    try:
        rclpy.spin(rtsp_stream_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        rtsp_stream_publisher.on_shutdown()
        rtsp_stream_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()