#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# 【重要】导入MultiThreadedExecutor
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
from collections import deque

# VideoStream 类和 RtspStreamPublisher 类的定义保持不变
# ... (请保留之前版本中这两个类的完整代码) ...

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
    def __init__(self):
        super().__init__('rtsp_stream_publisher')
        
        self.declare_parameter('rtsp_url', 'rtsp://192.168.1.10/color')
        self.declare_parameter('topic_name', '/kinova_camera/color/image_raw')
        self.declare_parameter('publish_rate', 30.0)

        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.get_logger().info(f"Connecting to RTSP URL: {rtsp_url}")
        self.get_logger().info(f"Publishing to topic: {topic_name}")
        self.get_logger().info(f"Publish rate: {publish_rate} Hz")

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        
        self.ros_image_queue = deque(maxlen=1)
        self.processing_stopped = False

        try:
            self.video_stream = VideoStream(rtsp_url)
            time.sleep(2.0)
        except IOError as e:
            self.get_logger().error(f'Failed to start video stream: {e}')
            # In ROS2, simply returning is cleaner than calling shutdown here
            raise e

        self.processing_thread = threading.Thread(target=self.frame_processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
            
        self.timer = self.create_timer(1.0 / publish_rate, self.publisher_callback)
        self.get_logger().info('RTSP stream publisher node has been started.')

    def frame_processing_thread(self):
        self.get_logger().info("Frame processing thread started.")
        while not self.processing_stopped:
            frame = self.video_stream.read()
            if frame is not None:
                ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.ros_image_queue.append(ros_image_msg)
            else:
                time.sleep(0.01)
        self.get_logger().info("Frame processing thread stopped.")

    def publisher_callback(self):
        if len(self.ros_image_queue) > 0:
            ros_image_msg = self.ros_image_queue.popleft()
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            ros_image_msg.header.frame_id = "kinova_camera_link"
            self.publisher_.publish(ros_image_msg)

    def on_shutdown(self):
        self.get_logger().info('Shutting down...')
        self.processing_stopped = True
        if hasattr(self, 'processing_thread'):
             self.processing_thread.join()
        if hasattr(self, 'video_stream'):
            self.video_stream.stop()


def main(args=None):
    rclpy.init(args=args)
    try:
        rtsp_stream_publisher = RtspStreamPublisher()

        # 【核心修改】使用MultiThreadedExecutor
        # 我们可以指定线程池的大小，num_threads=4 表示最多可以有4个回调并行执行
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(rtsp_stream_publisher)

        try:
            # 使用 executor.spin() 而不是 rclpy.spin()
            executor.spin()
        finally:
            # 确保在退出时正确关闭executor
            executor.shutdown()
            rtsp_stream_publisher.on_shutdown()
            rtsp_stream_publisher.destroy_node()

    except (IOError, KeyboardInterrupt) as e:
         # Handle exceptions gracefully
         if isinstance(e, IOError):
             rclpy.get_logger('main').error(f"Initialization failed: {e}")
         else:
             rclpy.get_logger('main').info("Shutdown requested by user.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()