'''
尝试使用抽帧
'''

import cv2
import time

# --- 1. 配置 ---
BASE_IP_ADDRESS = "192.168.1.10"
rtsp_url_color = f"rtsp://{BASE_IP_ADDRESS}/color"

# --- 2. 连接到视频流 ---
print(f"正在尝试连接颜色流: {rtsp_url_color}")
cap = cv2.VideoCapture(rtsp_url_color)

time.sleep(0.5)

if not cap.isOpened():
    print("!!! 致命错误: 无法打开颜色流。")
    exit()

print("\n✅ 连接成功！正在通过抽帧方式显示低延迟视频流... 按 'q' 键退出。")

# --- 3. 定义抽帧间隔 ---
# 这个值决定了我们每次要丢弃多少帧。
# 值越大，清空缓冲区的效果越好，但CPU消耗也可能略微增加。
# 通常 5 到 10 是一个不错的起点。
# 你也可以把它设置为0，看看高延迟的效果。
FRAMES_TO_SKIP = 5 

try:
    frame_counter = 0
    while True:
        # --- 核心抽帧逻辑 ---
        # 连续读取并丢弃指定数量的帧，以清空缓冲区
        for _ in range(FRAMES_TO_SKIP):
            cap.grab() # .grab() 比 .read() 更快，因为它只抓取帧，不解码

        # 现在读取我们真正想要显示的那一帧
        ret, frame = cap.read()

        # 检查是否成功读取
        if not ret:
            print("警告: 无法读取帧。")
            time.sleep(0.5)
            continue
        
        # --- 4. 显示图像 ---
        # 可以在这里添加一个简单的帧率计数器来观察效果
        frame_counter += 1
        # cv2.putText(frame, f"Frame: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Low Latency via Frame Skipping', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- 5. 释放资源 ---
    print("\n正在关闭数据流并释放资源...")
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")