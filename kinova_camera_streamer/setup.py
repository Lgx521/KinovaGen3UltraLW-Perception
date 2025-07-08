import os
from glob import glob
from setuptools import setup

package_name = 'kinova_camera_streamer'

setup(
    name=package_name,
    version='1.0.0', # 可以更新一下版本号，表明这是一个重构后的版本
    packages=[package_name],
    
    # data_files 确保 launch 文件和 package.xml 被正确安装
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 这一行会找到launch目录下的所有以.launch.py结尾的文件并安装它们
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    
    # install_requires 列出运行所需的非ROS Python包
    # 我们的代码需要 setuptools, opencv-python, numpy
    # 注意：rclpy, sensor_msgs, cv_bridge 是ROS的核心依赖，通常不需要在这里列出
    install_requires=['setuptools'], 
    
    zip_safe=True,
    
    # 包的元数据
    maintainer='Your Name', # 替换成你的名字
    maintainer_email='your.email@example.com', # 替换成你的邮箱
    description='A ROS2 package to stream video from an RTSP source using a low-latency GStreamer pipeline.',
    license='Apache License 2.0', # 或者你选择的其他开源许可证
    
    # 测试依赖
    tests_require=['pytest'],
    
    # entry_points 定义了 'ros2 run' 和 'ros2 launch' 可以找到的可执行脚本
    entry_points={
        'console_scripts': [
            # 格式: 'executable_name = package_name.python_module_name:main_function'
            # 我们将可执行脚本命名为 'gstreamer_node'
            'gstreamer_node = kinova_camera_streamer.rtsp_stream_node:main',
        ],
    },
)