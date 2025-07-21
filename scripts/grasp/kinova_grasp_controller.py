#!/usr/bin/env python3

import numpy as np
import time
from typing import List, Tuple, Optional, Dict

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_srvs.srv import Trigger
from moveit_msgs.msg import Constraints, JointConstraint, PositionIKRequest
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation as R

# Import custom service definitions
from kinova_graspnet_ros2.srv import DetectGrasps, ExecuteGrasp


class KinovaGraspController(Node):
    """ROS2 controller for executing grasps with Kinova Gen3 6DOF arm"""
    
    def __init__(self):
        super().__init__('kinova_grasp_controller')
        
        # Declare parameters
        self.declare_parameter('planning_group', 'manipulator')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'robotiq_85_base_link')
        self.declare_parameter('gripper_palm_frame', 'gripper_palm_center')
        self.declare_parameter('rgb_camera_frame', 'camera_color_frame')
        self.declare_parameter('depth_camera_frame', 'camera_depth_frame')
        self.declare_parameter('camera_frame', 'camera_depth_frame')
        self.declare_parameter('gripper_closed_position', 0.8)
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('approach_distance', 0.1)
        self.declare_parameter('retreat_distance', 0.1)
        self.declare_parameter('use_simplified_grasp', True)
        
        # Get parameters
        self.planning_group = self.get_parameter('planning_group').value
        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('ee_frame').value
        self.gripper_palm_frame = self.get_parameter('gripper_palm_frame').value
        self.rgb_camera_frame = self.get_parameter('rgb_camera_frame').value
        self.depth_camera_frame = self.get_parameter('depth_camera_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.gripper_closed_pos = self.get_parameter('gripper_closed_position').value
        self.gripper_open_pos = self.get_parameter('gripper_open_position').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.retreat_distance = self.get_parameter('retreat_distance').value
        self.use_simplified_grasp = self.get_parameter('use_simplified_grasp').value
        
        # Joint names for Kinova Gen3 6DOF
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6'
        ]
        self.gripper_joint_name = 'finger_joint'
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize callback group for parallel execution
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service clients
        self.grasp_detection_client = self.create_client(
            DetectGrasps,
            'detect_grasps',
            callback_group=self.callback_group
        )
        
        # MoveGroup action client
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self.callback_group
        )
        
        # IK service client
        self.ik_client = self.create_client(
            GetPositionIK,
            'compute_ik',
            callback_group=self.callback_group
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Publishers
        self.gripper_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/finger_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Services
        self.execute_grasp_srv = self.create_service(
            ExecuteGrasp,
            'execute_grasp',
            self.execute_grasp_callback,
            callback_group=self.callback_group
        )
        
        self.auto_grasp_srv = self.create_service(
            Trigger,
            'auto_grasp',
            self.auto_grasp_callback,
            callback_group=self.callback_group
        )
        
        # Transform test service
        self.test_transform_srv = self.create_service(
            Trigger,
            'test_transforms',
            self.test_transforms_callback,
            callback_group=self.callback_group
        )
        
        # State variables
        self.current_joint_state = None
        
        # Wait for services
        self.get_logger().info('Waiting for required services...')
        self.wait_for_services()
        
        self.get_logger().info('Kinova grasp controller initialized')
    
    def wait_for_services(self):
        """Wait for required services to be available"""
        services_to_wait = [
            (self.grasp_detection_client, 'grasp detection'),
            (self.ik_client, 'IK service')
        ]
        
        for client, name in services_to_wait:
            if not client.wait_for_service(timeout_sec=10.0):
                self.get_logger().warn(f'{name} service not available')
        
        # Wait for MoveGroup action
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('MoveGroup action server not available')
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state updates"""
        self.current_joint_state = msg
    
    #target_frame is "base_frame"
    def transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform pose to target frame"""
        try:
            self.get_logger().info(f'Transforming pose from {pose.header.frame_id} to {target_frame}')
            self.get_logger().info(f'Original pose: position=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})')
            
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                Time(),
                timeout=Duration(seconds=1.0)
            )
            
            transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(pose, transform)
            
            self.get_logger().info(f'Transformed pose: position=({transformed_pose.pose.position.x:.3f}, {transformed_pose.pose.position.y:.3f}, {transformed_pose.pose.position.z:.3f})')
            
            return transformed_pose
            
        except TransformException as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
    
    def get_transform_matrix(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        """Get transformation matrix from source_frame to target_frame
        
        Usage example:
        # Get transform from camera to base: ros2 run tf2_ros tf2_echo base_link camera_depth_frame
        transform_matrix = self.get_transform_matrix('base_link', 'camera_depth_frame')
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, 
                Time(), timeout=Duration(seconds=1.0)
            )
            
            # Extract translation
            trans = transform.transform.translation
            translation = np.array([trans.x, trans.y, trans.z])
            
            # Extract rotation (quaternion to rotation matrix)
            rot = transform.transform.rotation
            rotation_matrix = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = translation
            
            self.get_logger().info(f'Transform from {source_frame} to {target_frame}:')
            self.get_logger().info(f'Translation: {translation}')
            self.get_logger().info(f'Rotation matrix:\n{rotation_matrix}')
            
            return transform_matrix
            
        except TransformException as ex:
            self.get_logger().warn(f'Could not get transform from {source_frame} to {target_frame}: {ex}')
            return None
    
    def compute_ik(self, target_pose: PoseStamped) -> Optional[List[float]]:
        """Compute inverse kinematics for target pose"""
        if not self.current_joint_state:
            self.get_logger().error('No joint state available')
            return None
        
        # Create IK request
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.planning_group
        ik_request.ik_request.robot_state.joint_state = self.current_joint_state
        ik_request.ik_request.pose_stamped = target_pose
        ik_request.ik_request.ik_link_name = self.ee_frame  # Use configured end effector frame
        ik_request.ik_request.timeout.sec = 5
        # Note: attempts attribute may not exist in this version of MoveIt
        
        # Call IK service
        future = self.ik_client.call_async(ik_request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is None:
            self.get_logger().error('IK service call failed')
            return None
        
        response = future.result()
        
        if response.error_code.val != response.error_code.SUCCESS:
            self.get_logger().error(f'IK failed with error code: {response.error_code.val}')
            self.get_logger().error(f'Target pose: position=({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f}, {target_pose.pose.position.z:.3f})')
            self.get_logger().error(f'Target pose frame: {target_pose.header.frame_id}')
            return None
        
        # Extract joint values
        joint_values = []
        for joint_name in self.joint_names:
            if joint_name in response.solution.joint_state.name:
                idx = response.solution.joint_state.name.index(joint_name)
                joint_values.append(response.solution.joint_state.position[idx])
        
        return joint_values
    
    def move_to_joint_positions(self, joint_positions: List[float],
                               max_velocity_scaling: float = 0.3,
                               max_acceleration_scaling: float = 0.3) -> bool:
        """Move arm to joint positions using MoveGroup"""
        if len(joint_positions) != len(self.joint_names):
            self.get_logger().error(f'Expected {len(self.joint_names)} joint positions, got {len(joint_positions)}')
            return False
        
        # Create MoveGroup goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.planning_group
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = max_velocity_scaling
        goal_msg.request.max_acceleration_scaling_factor = max_acceleration_scaling
        
        # Create joint constraints
        constraints = Constraints()
        for joint_name, position in zip(self.joint_names, joint_positions):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        
        goal_msg.request.goal_constraints = [constraints]
        
        # Send goal
        self.get_logger().info('Sending movement goal to MoveGroup')
        goal_future = self._move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)
        
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Movement goal rejected')
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        if result.result.error_code.val == 1:  # SUCCESS
            self.get_logger().info('Movement completed successfully')
            return True
        else:
            self.get_logger().error(f'Movement failed with error code: {result.result.error_code.val}')
            return False
    
    def move_to_pose(self, target_pose: PoseStamped,
                     max_velocity_scaling: float = 0.3,
                     max_acceleration_scaling: float = 0.3) -> bool:
        """Move end effector to target pose"""
        # Compute IK
        joint_positions = self.compute_ik(target_pose)
        if joint_positions is None:
            return False
        
        # Execute movement
        return self.move_to_joint_positions(
            joint_positions,
            max_velocity_scaling,
            max_acceleration_scaling
        )
    
    def control_gripper(self, position: float, max_effort: float = 100.0) -> bool:
        """Control gripper position"""
        trajectory = JointTrajectory()
        trajectory.joint_names = [self.gripper_joint_name]
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.velocities = [0.0]
        point.effort = [max_effort]
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0
        
        trajectory.points = [point]
        
        self.gripper_cmd_pub.publish(trajectory)
        
        # Wait for gripper to reach position
        time.sleep(1.5)
        
        return True
    
    def compute_approach_pose(self, grasp_pose: PoseStamped, distance: float) -> PoseStamped:
        """Compute approach pose along grasp approach vector"""
        
        approach_pose = PoseStamped()
        approach_pose.header = grasp_pose.header
        
        # Extract rotation from grasp pose
        q = grasp_pose.pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rotation.as_matrix()
        
        # Approach vector is along positive Z axis of grasp frame
        # In TF camera frame, Z-axis points forward (approach direction)
        approach_vector = rotation_matrix[:, 2]
        
        # Compute approach position
        approach_pose.pose.position.x = grasp_pose.pose.position.x + approach_vector[0] * distance
        approach_pose.pose.position.y = grasp_pose.pose.position.y + approach_vector[1] * distance
        approach_pose.pose.position.z = grasp_pose.pose.position.z + approach_vector[2] * distance
        
        # Keep same orientation
        approach_pose.pose.orientation = grasp_pose.pose.orientation
        
        return approach_pose
    
    def transform_grasp_center_to_ee(self, grasp_pose_camera: PoseStamped) -> PoseStamped:
        """
        Transform grasp pose from camera frame to end-effector frame
        
        This function handles the transformation from GraspNet's output (grasp_center in camera frame)
        to the robot's end-effector pose in base_link frame.
        
        Args:
            grasp_pose_camera: Grasp center pose in camera frame from GraspNet
            
        Returns:
            End-effector pose in base_link frame for robot execution
        """
        try:
            # 💾 保存所有变换矩阵用于调试
            import json
            import time as time_module
            
            # Step 1: Transform grasp_center from camera frame to base_link frame
            grasp_center_base = self.transform_pose(grasp_pose_camera, self.base_frame)
            if grasp_center_base is None:
                self.get_logger().error("Failed to transform grasp center to base frame")
                return None
            
            # 💾 获取并保存关键变换矩阵
            debug_matrices = {}
            timestamp = int(time_module.time())
            
            # 获取所有相关的变换矩阵
            transform_pairs = [
                ('camera_depth_frame', 'grasp_center', 'grasp_center_to_camera'),
                (self.ee_frame, 'camera_depth_frame', 'camera_to_ee'),  
                (self.base_frame, self.ee_frame, 'ee_to_base'),
                (self.base_frame, 'grasp_center', 'grasp_center_to_base_direct'),
                (self.base_frame, 'camera_depth_frame', 'camera_to_base_direct'),
                (self.ee_frame, 'grasp_center', 'grasp_center_to_ee_direct')
            ]
            
            for target_frame, source_frame, name in transform_pairs:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame, source_frame, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    matrix = self.transform_to_matrix(transform)
                    debug_matrices[name] = {
                        'matrix': matrix.tolist(),
                        'translation': [float(matrix[0,3]), float(matrix[1,3]), float(matrix[2,3])],
                        'target_frame': target_frame,
                        'source_frame': source_frame
                    }
                    self.get_logger().info(f"✅ 获取变换 {source_frame} → {target_frame}")
                except Exception as e:
                    debug_matrices[name] = {'error': str(e)}
                    self.get_logger().warn(f"❌ 无法获取变换 {source_frame} → {target_frame}: {e}")
            
            # 保存输入的抓取位姿
            debug_matrices['input_grasp_pose_camera'] = {
                'position': [float(grasp_pose_camera.pose.position.x), 
                           float(grasp_pose_camera.pose.position.y),
                           float(grasp_pose_camera.pose.position.z)],
                'orientation': [float(grasp_pose_camera.pose.orientation.x),
                              float(grasp_pose_camera.pose.orientation.y), 
                              float(grasp_pose_camera.pose.orientation.z),
                              float(grasp_pose_camera.pose.orientation.w)],
                'frame_id': grasp_pose_camera.header.frame_id
            }
            
            # Step 2: Get current transform from grasp_center to end-effector
            # This tells us the relative pose between grasp_center and ee
            try:
                transform_gc_to_ee = self.tf_buffer.lookup_transform(
                    self.ee_frame,
                    'grasp_center',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                
                # Convert transform to matrix
                gc_to_ee_matrix = self.transform_to_matrix(transform_gc_to_ee)
                
                self.get_logger().info(f"✅ 成功获取TF变换 grasp_center → {self.ee_frame}")
                self.get_logger().info(f"  Translation: ({gc_to_ee_matrix[0,3]:.6f}, {gc_to_ee_matrix[1,3]:.6f}, {gc_to_ee_matrix[2,3]:.6f})")
                
                debug_matrices['used_gc_to_ee'] = {
                    'source': 'TF_lookup',
                    'matrix': gc_to_ee_matrix.tolist()
                }
                
            except TransformException as e:
                # If grasp_center frame not available, use default offset
                self.get_logger().warn(f"❌ grasp_center frame not available: {e}")
                self.get_logger().warn("使用硬编码偏移 -0.129m")
                # Default offset: ee is ~8.1cm behind grasp_center along approach direction
                gc_to_ee_matrix = np.eye(4)
                gc_to_ee_matrix[2, 3] = -0.129  # Offset along Z axis
                
                debug_matrices['used_gc_to_ee'] = {
                    'source': 'hardcoded_fallback',
                    'matrix': gc_to_ee_matrix.tolist(),
                    'error': str(e)
                }
            
            # Step 3: Convert grasp_center pose to matrix
            grasp_center_matrix = self.pose_stamped_to_matrix(grasp_center_base)
            
            # Step 4: Compute end-effector pose in base frame
            # ee_in_base = grasp_center_in_base * grasp_center_to_ee
            ee_pose_matrix = grasp_center_matrix @ gc_to_ee_matrix
            
            # Step 5: Convert back to PoseStamped
            ee_pose_base = self.matrix_to_pose_stamped(ee_pose_matrix, self.base_frame)
            
            # 💾 保存计算过程和最终结果
            debug_matrices['computed_grasp_center_base'] = {
                'matrix': grasp_center_matrix.tolist(),
                'position': [float(grasp_center_base.pose.position.x),
                           float(grasp_center_base.pose.position.y), 
                           float(grasp_center_base.pose.position.z)],
                'orientation': [float(grasp_center_base.pose.orientation.x),
                              float(grasp_center_base.pose.orientation.y),
                              float(grasp_center_base.pose.orientation.z), 
                              float(grasp_center_base.pose.orientation.w)]
            }
            
            debug_matrices['computed_ee_base'] = {
                'matrix': ee_pose_matrix.tolist(), 
                'position': [float(ee_pose_base.pose.position.x),
                           float(ee_pose_base.pose.position.y),
                           float(ee_pose_base.pose.position.z)],
                'orientation': [float(ee_pose_base.pose.orientation.x),
                              float(ee_pose_base.pose.orientation.y),
                              float(ee_pose_base.pose.orientation.z),
                              float(ee_pose_base.pose.orientation.w)]
            }
            
            # 保存到文件
            debug_file = f"/home/q/Graspnet/graspnet-baseline/kinova_graspnet_ros2/grasp_debug_{timestamp}.json"
            try:
                with open(debug_file, 'w') as f:
                    json.dump(debug_matrices, f, indent=2)
                self.get_logger().info(f"💾 调试信息已保存到: {debug_file}")
            except Exception as save_error:
                self.get_logger().warn(f"❌ 无法保存调试文件: {save_error}")
            
            self.get_logger().info(f'Grasp transformation complete:')
            self.get_logger().info(f'  Grasp center (camera): ({grasp_pose_camera.pose.position.x:.3f}, {grasp_pose_camera.pose.position.y:.3f}, {grasp_pose_camera.pose.position.z:.3f})')
            self.get_logger().info(f'  Grasp center (base): ({grasp_center_base.pose.position.x:.3f}, {grasp_center_base.pose.position.y:.3f}, {grasp_center_base.pose.position.z:.3f})')
            self.get_logger().info(f'  End-effector (base): ({ee_pose_base.pose.position.x:.3f}, {ee_pose_base.pose.position.y:.3f}, {ee_pose_base.pose.position.z:.3f})')
            
            return ee_pose_base
            
        except Exception as e:
            self.get_logger().error(f'Failed to transform grasp: {str(e)}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')
            return None
    
    def transform_to_matrix(self, transform: tf2_ros.TransformStamped) -> np.ndarray:
        """Convert TransformStamped to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Translation
        t = transform.transform.translation
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z
        
        # Rotation
        q = transform.transform.rotation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix
    
    def pose_stamped_to_matrix(self, pose: PoseStamped) -> np.ndarray:
        """Convert PoseStamped to 4x4 transformation matrix"""
        matrix = np.eye(4)
        
        # Translation
        matrix[0, 3] = pose.pose.position.x
        matrix[1, 3] = pose.pose.position.y
        matrix[2, 3] = pose.pose.position.z
        
        # Rotation
        q = pose.pose.orientation
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix[:3, :3] = rotation.as_matrix()
        
        return matrix
    
    def matrix_to_pose_stamped(self, matrix: np.ndarray, frame_id: str) -> PoseStamped:
        """Convert 4x4 transformation matrix to PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        
        # Translation
        pose.pose.position.x = float(matrix[0, 3])
        pose.pose.position.y = float(matrix[1, 3])
        pose.pose.position.z = float(matrix[2, 3])
        
        # Rotation
        rotation = R.from_matrix(matrix[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        
        return pose
    
    def execute_grasp_sequence(self, grasp_pose: PoseStamped, grasp_width: float,
                              approach_distance: float,
                              max_velocity_scaling: float = 0.3,
                              max_acceleration_scaling: float = 0.3) -> Tuple[bool, str]:
        """Execute simplified grasp sequence"""
        try:
            # Transform grasp pose from camera frame to end-effector pose in base frame
            # This handles the conversion from grasp_center to end-effector
            ee_pose_base = self.transform_grasp_center_to_ee(grasp_pose)
            if ee_pose_base is None:
                return False, "Failed to transform grasp pose to end-effector frame"
            
            # 1. Open gripper
            self.get_logger().info('Opening gripper')
            self.control_gripper(self.gripper_open_pos)
            
            # 2. Move directly to grasp pose (simplified - no approach phase)
            self.get_logger().info('Moving directly to grasp pose')
            if not self.move_to_pose(ee_pose_base, max_velocity_scaling * 0.5, max_acceleration_scaling * 0.5):
                return False, "Failed to reach grasp pose"
            
            # 3. Close gripper
            self.get_logger().info(f'Closing gripper to width: {grasp_width}')
            # Convert grasp width to gripper position (this mapping depends on your gripper)
            gripper_position = max(self.gripper_closed_pos * (1 - grasp_width / 0.085), 0.0)
            self.control_gripper(gripper_position)
            
            return True, "Simplified grasp executed successfully"
            
        except Exception as e:
            self.get_logger().error(f'Grasp execution failed: {str(e)}')
            import traceback
            self.get_logger().error(f'Full traceback: {traceback.format_exc()}')
            return False, f"Exception during grasp execution: {str(e)}"
    
    def execute_grasp_sequence_with_approach(self, grasp_pose: PoseStamped, grasp_width: float,
                              approach_distance: float,
                              max_velocity_scaling: float = 0.3,
                              max_acceleration_scaling: float = 0.3) -> Tuple[bool, str]:
        """Execute complete grasp sequence with approach and retreat"""
        try:
            # Transform grasp pose from camera frame to end-effector pose in base frame
            # This handles the conversion from grasp_center to end-effector
            ee_pose_base = self.transform_grasp_center_to_ee(grasp_pose)
            if ee_pose_base is None:
                return False, "Failed to transform grasp pose to end-effector frame"
            
            # 1. Open gripper
            self.get_logger().info('Opening gripper')
            self.control_gripper(self.gripper_open_pos)
            
            # 2. Move to approach pose 
            approach_pose = self.compute_approach_pose(ee_pose_base, approach_distance)
            
            if not self.move_to_pose(approach_pose, max_velocity_scaling, max_acceleration_scaling):
                return False, "Failed to reach approach pose"
            
            # 3. Move to grasp pose
            self.get_logger().info('Moving to grasp pose')
            if not self.move_to_pose(ee_pose_base, max_velocity_scaling * 0.5, max_acceleration_scaling * 0.5):
                return False, "Failed to reach grasp pose"
            
            # 4. Close gripper
            self.get_logger().info(f'Closing gripper to width: {grasp_width}')
            # Convert grasp width to gripper position (this mapping depends on your gripper)
            gripper_position = max(self.gripper_closed_pos * (1 - grasp_width / 0.085), 0.0)
            self.control_gripper(gripper_position)
            
            # 5. Retreat
            retreat_pose = self.compute_approach_pose(ee_pose_base, self.retreat_distance)
            self.get_logger().info('Retreating')
            if not self.move_to_pose(retreat_pose, max_velocity_scaling, max_acceleration_scaling):
                self.get_logger().warn('Failed to retreat, but grasp may still be successful')
            
            return True, "Grasp executed successfully"
            
        except Exception as e:
            self.get_logger().error(f'Grasp execution failed: {str(e)}')
            import traceback
            self.get_logger().error(f'Full traceback: {traceback.format_exc()}')
            return False, f"Exception during grasp execution: {str(e)}"
    
    def execute_grasp_callback(self, request: ExecuteGrasp.Request, response: ExecuteGrasp.Response):
        """Service callback for executing a grasp"""
        self.get_logger().info('Received grasp execution request')
        
        start_time = time.time()
        
        # Choose execution method based on parameter
        if self.use_simplified_grasp:
            self.get_logger().info('Using simplified grasp execution (direct movement)')
            success, message = self.execute_grasp_sequence(
                request.grasp_pose,
                request.grasp_width,
                request.approach_distance,
                request.max_velocity_scaling,
                request.max_acceleration_scaling
            )
        else:
            self.get_logger().info('Using full grasp execution (with approach and retreat)')
            success, message = self.execute_grasp_sequence_with_approach(
                request.grasp_pose,
                request.grasp_width,
                request.approach_distance,
                request.max_velocity_scaling,
                request.max_acceleration_scaling
            )
        
        response.success = success
        response.message = message
        response.execution_time = time.time() - start_time
        
        # Get final pose
        if self.current_joint_state:
            # Could compute forward kinematics here to get final pose
            # For now, just copy the requested pose
            response.final_pose = request.grasp_pose
        
        return response
    
    def auto_grasp_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service callback for automatic grasp detection and execution"""
        self.get_logger().info('Auto grasp requested - this would trigger camera capture and grasp detection')
        
        # This is a placeholder - in a real implementation, you would:
        # 1. Capture current camera images
        # 2. Call grasp detection service
        # 3. Select best grasp
        # 4. Execute grasp
        
        response.success = True
        response.message = "Auto grasp functionality not fully implemented yet"
        
        return response
    
    def test_transforms_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Service callback to test coordinate transforms"""
        self.get_logger().info('Testing coordinate transforms...')
        
        transform_tests = [
            (self.base_frame, self.depth_camera_frame, "Camera depth to base"),
            (self.base_frame, self.rgb_camera_frame, "Camera RGB to base"), 
            (self.depth_camera_frame, self.rgb_camera_frame, "RGB to depth camera"),
            (self.base_frame, self.ee_frame, "Gripper mount to base"),
            (self.base_frame, self.gripper_palm_frame, "Gripper palm to base"),
            (self.ee_frame, self.gripper_palm_frame, "Gripper palm to mount"),
            (self.gripper_palm_frame, self.depth_camera_frame, "Camera depth to gripper palm"),
            (self.gripper_palm_frame, self.rgb_camera_frame, "Camera RGB to gripper palm")
        ]
        
        success_count = 0
        total_tests = len(transform_tests)
        messages = []
        
        try:
            for target_frame, source_frame, description in transform_tests:
                self.get_logger().info(f'=== Testing {description} ===')
                transform_matrix = self.get_transform_matrix(target_frame, source_frame)
                
                if transform_matrix is not None:
                    success_count += 1
                    messages.append(f"✓ {description}: {source_frame} -> {target_frame}")
                else:
                    messages.append(f"✗ {description}: Failed to get transform {source_frame} -> {target_frame}")
            
            response.success = success_count == total_tests
            response.message = f"Transform test results ({success_count}/{total_tests} successful):\n" + "\n".join(messages)
            
            if success_count == total_tests:
                self.get_logger().info("All coordinate transforms are available!")
            else:
                self.get_logger().warn(f"Only {success_count}/{total_tests} transforms are available")
                
        except Exception as e:
            response.success = False
            response.message = f"Error testing transforms: {str(e)}"
            self.get_logger().error(f"Transform test failed: {e}")
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = KinovaGraspController()
        
        # Use MultiThreadedExecutor for handling callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()