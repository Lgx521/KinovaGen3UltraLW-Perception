import cv2
import time
import threading

# --- 1. 创建一个独立的视频流读取类 ---
# 这个类会在一个单独的线程中运行，以确保我们总是能获取到最新的帧
class VideoStream:
    """A class to read frames from a camera in a separate thread."""
    
    def __init__(self, src=0):
        # 初始化视频流
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"!!! 致命错误: 无法打开视频流: {src}")
            raise IOError("Cannot open video stream")
            
        # 读取第一帧来获取流的状态
        (self.grabbed, self.frame) = self.stream.read()
        
        # 标志位，用于停止线程
        self.stopped = False
        
        # 启动线程来执行 update 方法
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # 设置为守护线程，主程序退出时线程也退出
        self.thread.start()
        print(f"✅ 视频流已在后台线程启动: {src}")

    def update(self):
        # 这是线程的主体，一个高速循环，不断读取帧
        while not self.stopped:
            # 如果流中有新的帧，就读取它
            (self.grabbed, self.frame) = self.stream.read()
        # 当循环结束时，释放视频流资源
        self.stream.release()

    def read(self):
        # 返回当前最新的一帧
        return self.frame

    def stop(self):
        # 设置标志位来停止线程
        self.stopped = True
        # 等待线程完全结束
        self.thread.join()

# --- 2. 主程序 ---
if __name__ == '__main__':
    # 配置RTSP URL
    BASE_IP_ADDRESS = "192.168.1.10"
    rtsp_url_color = f"rtsp://{BASE_IP_ADDRESS}/color"

    # 使用我们创建的 VideoStream 类来实例化视频流
    # 这会自动在后台启动读取
    try:
        vs = VideoStream(rtsp_url_color)
    except IOError as e:
        print(e)
        exit()
        
    # 给视频流一点时间来“预热”和填充第一帧
    time.sleep(2.0)
    
    print("\n✅ 主程序已启动，正在显示实时视频流... 按 'q' 键退出。")

    try:
        while True:
            # 从后台线程获取最新的帧，这几乎是瞬时的
            frame = vs.read()
            
            # 如果帧是None（可能在启动初期或流断开时），则跳过
            if frame is None:
                continue

            # 在这里可以添加您的图像处理逻辑
            # ...

            # 显示图像
            cv2.imshow("Real-time Color Stream (Low Latency)", frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 确保在退出时停止后台线程并关闭所有窗口
        print("\n正在停止后台线程并释放资源...")
        vs.stop()
        cv2.destroyAllWindows()
        print("程序已退出。")