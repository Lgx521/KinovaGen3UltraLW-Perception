import rclpy
from rclpy.action import ActionServer, ActionClient, GoalResponse
from rclpy.node import Node
import time

# 导入我们自己定义的Action
from arm_controller.action import MoveToPose

# 导入底层MoveGroup Action及其消息
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped


class PoseActionServer(Node):
    """
    This node provides a simple Action interface (/move_to_pose) to control the robot arm.
    It receives a simple PoseStamped goal and translates it into a complex goal for the
    underlying /move_group Action server provided by MoveIt.
    """

    def __init__(self):
        super().__init__('pose_action_server')
        
        # 1. 创建我们自己的Action服务器，等待外部调用
        self._action_server = ActionServer(
            self,
            MoveToPose,
            'move_to_pose', # 我们的Action的名称
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback)
        self.get_logger().info('MoveToPose Action Server has been started.')

        # 2. 创建用于与MoveGroup通信的Action客户端
        self._move_group_client = ActionClient(self, MoveGroup, '/move_group')
        self.get_logger().info('Connecting to /move_group action server...')
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/move_group action server not available after 5s! Shutting down.')
            self.destroy_node()
            return
        self.get_logger().info('/move_group action server is available.')

    def goal_callback(self, goal_request):
        """Accepts or rejects a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Executes a goal."""
        self.get_logger().info('Executing goal...')
        
        # 从收到的goal中提取目标姿态
        target_pose = goal_handle.request.target_pose

        # 构建发送给 /move_group 的复杂Goal
        move_group_goal = self.create_move_group_goal(target_pose)
        
        # 发送feedback给我们的客户端
        feedback_msg = MoveToPose.Feedback()
        feedback_msg.status = "PLANNING"
        goal_handle.publish_feedback(feedback_msg)

        # 异步发送Goal给MoveGroup
        goal_handle_movegroup = await self._move_group_client.send_goal_async(move_group_goal)
        
        if not goal_handle_movegroup.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            goal_handle.abort()
            result = MoveToPose.Result()
            result.success = False
            result.message = "MoveGroup goal rejected"
            return result

        self.get_logger().info('MoveGroup goal accepted.')

        # 获取最终结果
        result_from_movegroup = await goal_handle_movegroup.get_result_async()
        
        final_result = MoveToPose.Result()
        if result_from_movegroup.result.error_code.val == 1: # SUCCESS
            self.get_logger().info('MoveGroup motion successful!')
            final_result.success = True
            final_result.message = "Motion successful"
            goal_handle.succeed()
        else:
            error_message = f"MoveGroup motion failed with error code {result_from_movegroup.result.error_code.val}"
            self.get_logger().error(error_message)
            final_result.success = False
            final_result.message = error_message
            goal_handle.abort()
            
        return final_result

    def create_move_group_goal(self, pose_stamped: PoseStamped) -> MoveGroup.Goal:
        """Helper function to convert a simple PoseStamped to a complex MoveGroup.Goal"""
        goal_msg = MoveGroup.Goal()
        plan_request = MotionPlanRequest()
        
        # =================================================================
        # TODO: 在这里确认并修改为你的机器人配置
        # =================================================================
        GROUP_NAME = "arm_manipulator"  # 在RViz中确认的规划组名称
        END_EFFECTOR_LINK = "link_6"    # 在RViz中确认的末端连杆名称
        # =================================================================

        plan_request.group_name = GROUP_NAME
        plan_request.num_planning_attempts = 5
        plan_request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = pose_stamped.header.frame_id
        pos_constraint.link_name = END_EFFECTOR_LINK
        
        bounding_box = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01] # 1cm的容差
        bounding_box.primitives.append(primitive)
        bounding_box.primitive_poses.append(pose_stamped.pose)
        pos_constraint.constraint_region = bounding_box
        constraints.position_constraints.append(pos_constraint)
        
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = pose_stamped.header.frame_id
        orient_constraint.link_name = END_EFFECTOR_LINK
        orient_constraint.orientation = pose_stamped.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.1
        orient_constraint.absolute_y_axis_tolerance = 0.1
        orient_constraint.absolute_z_axis_tolerance = 0.1
        orient_constraint.weight = 1.0
        constraints.orientation_constraints.append(orient_constraint)
        
        plan_request.goal_constraints.append(constraints)
        
        goal_msg.request = plan_request
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 5
        
        return goal_msg


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PoseActionServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()