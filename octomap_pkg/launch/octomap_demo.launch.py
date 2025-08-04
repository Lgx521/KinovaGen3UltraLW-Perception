# my_robot_launch/launch/octomap.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('octomap_pkg'),
        'config',
        'octomap_config.yaml'
    )

    octomap_server_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('cloud_in', '/camera/depth/color/points')
        ]
    )
    

#    rviz_node = Node(
#        package='rviz2',
#        executable='rviz2',
#        name='rviz2',
#        arguments=['-d', os.path.join(get_package_share_directory('my_robot_config'), 'rviz', #'octomap.rviz')] 
    # )

    return LaunchDescription([
        octomap_server_node,
#        rviz_node
    ])
