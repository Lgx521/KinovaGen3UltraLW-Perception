#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, ActionClient, GoalResponse
from rclpy.node import Node
import time

from arm_action_controller.action import MoveToPose

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped

class PoseActionServer(Node):
    def __init__(self):
        super().__init__('pose_action_server')
        
        self._action_server = ActionServer(
            self,
            MoveToPose,
            'move_to_pose',
            goal_callback=self.goal_callback,
            execute_callback=self.execute_callback)
        self.get_logger().info('MoveToPose Action Server has been started.')

        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.get_logger().info('Connecting to /move_group action server...')
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/move_group action server not available after 5s! Shutting down.')
            self.destroy_node()
            return
        self.get_logger().info('/move_group action server is available.')

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        target_pose = goal_handle.request.target_pose
        move_group_goal = self.create_move_group_goal(target_pose)
        
        feedback_msg = MoveToPose.Feedback()
        feedback_msg.status = "PLANNING"
        goal_handle.publish_feedback(feedback_msg)

        goal_handle_movegroup = await self._move_group_client.send_goal_async(move_group_goal)
        
        if not goal_handle_movegroup.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            goal_handle.abort()
            result = MoveToPose.Result()
            result.success = False
            result.message = "MoveGroup goal rejected"
            return result

        self.get_logger().info('MoveGroup goal accepted.')
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
        goal_msg = MoveGroup.Goal()
        plan_request = MotionPlanRequest()
        
        GROUP_NAME = "manipulator"
        END_EFFECTOR_LINK = "end_effector_link"

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
        primitive.dimensions = [0.01, 0.01, 0.01]
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