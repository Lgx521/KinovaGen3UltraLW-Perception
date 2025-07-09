from setuptools import setup

package_name = 'arm_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sgan',
    maintainer_email='sgan@todo.todo',
    description='A package to control the arm via a custom action interface.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 保留你之前的节点（如果有的话）
            # 'move_to_pose_node = arm_controller.move_to_pose:main',
            # 'move_via_action_node = arm_controller.move_via_action:main',
            # 添加我们的新Action服务器节点
            'pose_action_server = arm_controller.pose_action_server:main',
        ],
    },
)