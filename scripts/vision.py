import cv2
import time

# --- 1. 配置 ---
# 确认这是您相机的正确IP地址
BASE_IP_ADDRESS = "192.168.1.10"

# 构建颜色流的 RTSP URL
rtsp_url_color = f"rtsp://{BASE_IP_ADDRESS}/color"

print(f"正在尝试连接颜色流: {rtsp_url_color}")

# --- 2. 连接到视频流 ---
# 使用OpenCV的VideoCapture打开RTSP流 (使用默认后端)
cap_color = cv2.VideoCapture(rtsp_url_color)

# 增加一个小的延时，给网络流一些缓冲时间，提高稳定性
time.sleep(2.0)

# --- 3. 检查初始连接 ---
if not cap_color.isOpened():
    print("!!! 致命错误: 无法打开颜色流。")
    print("请检查:")
    print("1. URL是否正确: ", rtsp_url_color)
    print("2. 您的电脑和相机是否在同一个网络 (尝试ping)")
    print("3. 防火墙是否阻止了Python程序访问网络")
    exit()

print("\n✅ 连接成功！正在显示颜色视频流... 按 'q' 键退出。")

try:
    # --- 4. 循环读取并显示帧 ---
    while True:
        # 从流中读取一帧
        ret, frame = cap_color.read()

        # 如果ret为False，表示读取失败 (可能网络中断)
        if not ret:
            print("警告: 无法从颜色流中读取帧，可能已断开连接。正在尝试重新获取...")
            time.sleep(0.5) # 等待一下，看是否能恢复
            continue
        
        # --- 5. 显示图像 ---
        cv2.imshow('Color Stream', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # --- 6. 释放资源 ---
    print("\n正在关闭数据流并释放资源...")
    cap_color.release()
    cv2.destroyAllWindows()
    print("程序已退出。")