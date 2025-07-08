import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- 静态TF发布器节点 ---
    # 这个节点会发布一个从 'base_link' 到 'kinova_camera_link' 的静态坐标变换。
    # 你可以根据实际情况调整这里的父坐标系和子坐标系，以及它们之间的相对位置和姿态。
    #
    # 参数格式: x y z yaw pitch roll parent_frame_id child_frame_id
    # x, y, z:          以米为单位的平移
    # yaw, pitch, roll: 以弧度为单位的旋转
    #
    # 示例: 将相机放在base_link前方0.5米，向下看45度
    # x=0.5, y=0, z=0
    # yaw=0, pitch=0.785 (pi/4), roll=0
    static_tf_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_camera',
        output='screen',
        arguments=[
            '0.1', '0.0', '0.2',  # X, Y, Z (平移)
            '0', '0', '0',        # Yaw, Pitch, Roll (旋转)
            'base_link',          # 父坐标系 (Parent Frame)
            'kinova_camera_link'  # 子坐标系 (Child Frame, 必须与图像消息的frame_id匹配)
        ]
    )

    # --- 你的摄像头节点 ---
    # 我们也在这里启动你的摄像头节点
    camera_streamer_node = Node(
        package='kinova_camera_streamer',

        executable='gstreamer_node',
        name='kinova_camera_streamer_node',
        output='screen',
        # 【新增】现在可以方便地在这里配置参数
        parameters=[
            {'gstreamer_latency_ms': 0},
            {'frame_id': 'kinova_camera_link'}
        ]
    )

    # --- 返回启动描述 ---
    # LaunchDescription会收集所有要启动的节点并一起执行它们
    return LaunchDescription([
        static_tf_publisher_node,
        camera_streamer_node
    ])