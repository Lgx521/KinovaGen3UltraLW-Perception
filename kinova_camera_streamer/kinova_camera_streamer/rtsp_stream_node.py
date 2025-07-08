#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

class GStreamerPublisher(Node):
    """
    A ROS2 node that captures a video stream from an RTSP source using a 
    low-latency GStreamer pipeline and publishes it as a sensor_msgs/Image topic.
    """
    def __init__(self):
        super().__init__('gstreamer_publisher_node')
        
        # --- 1. 参数声明 ---
        self.declare_parameter('rtsp_url', 'rtsp://192.168.1.10/color')
        self.declare_parameter('topic_name', '/kinova_camera/color/image_raw')
        self.declare_parameter('frame_id', 'kinova_camera_link')
        self.declare_parameter('gstreamer_latency_ms', 0)

        # 获取参数
        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.frame_id_ = self.get_parameter('frame_id').get_parameter_value().string_value
        gstreamer_latency = self.get_parameter('gstreamer_latency_ms').get_parameter_value().integer_value
        
        # --- 2. 构建GStreamer管道 ---
        gstreamer_pipeline = (
            f'rtspsrc location={rtsp_url} latency={gstreamer_latency} ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            'videoconvert ! video/x-raw,format=BGR ! appsink drop=true'
        )
        
        self.get_logger().info(f"Using GStreamer pipeline: {gstreamer_pipeline}")

        # --- 3. 初始化VideoCapture ---
        self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("!!! Failed to open GStreamer pipeline.")
            raise IOError("Cannot open GStreamer pipeline")

        # --- 4. 设置ROS发布者 ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.publisher_ = self.create_publisher(Image, topic_name, qos_profile)
        self.bridge = CvBridge()
        
        # --- 5. 设置线程 ---
        self.shutdown_event = threading.Event()
        self.capture_thread = threading.Thread(target=self.run_capture_loop)
        self.capture_thread.daemon = True

        self.get_logger().info('GStreamer publisher node has been started.')

    def start_capture(self):
        """启动捕获线程"""
        self.capture_thread.start()

    def run_capture_loop(self):
        """后台线程，只负责读取帧并发布"""
        self.get_logger().info("Capture and publish thread started.")
        while not self.shutdown_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('GStreamer: Dropped frame or end of stream.', throttle_duration_sec=5)
                time.sleep(0.1)
                continue
            
            # 直接在这里转换和发布
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            ros_image_msg.header.frame_id = self.frame_id_
            self.publisher_.publish(ros_image_msg)
            
        self.cap.release()
        self.get_logger().info("Capture thread finished.")

    def on_shutdown(self):
        """节点关闭时的清理工作"""
        self.get_logger().info('Shutdown requested.')
        self.shutdown_event.set()
        self.capture_thread.join(timeout=2) # 等待线程结束

def main(args=None):
    rclpy.init(args=args)
    
    gstreamer_publisher_node = None
    try:
        gstreamer_publisher_node = GStreamerPublisher()
        gstreamer_publisher_node.start_capture()
        
        # 使用单线程执行器就足够了，因为所有耗时工作都在独立的capture_thread中
        rclpy.spin(gstreamer_publisher_node)

    except (IOError, KeyboardInterrupt) as e:
        if isinstance(e, IOError):
            rclpy.get_logger('main').error(f"Initialization failed: {e}")
        else: # KeyboardInterrupt
            rclpy.get_logger('main').info("Shutdown requested by user.")
    finally:
        if gstreamer_publisher_node:
            gstreamer_publisher_node.on_shutdown()
            gstreamer_publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()