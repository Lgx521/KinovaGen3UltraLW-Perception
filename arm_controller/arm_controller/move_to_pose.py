import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger

from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from geometry_msgs.msg import PoseStamped


def main():
    # 初始化rclpy和ROS 2节点
    rclpy.init()
    logger = get_logger("move_to_pose")
    node = Node("move_to_pose_node")

    # -------------------------------------------------------------------
    # 1. 初始化 MoveItPy
    # -------------------------------------------------------------------
    # MoveItPy是与MoveGroup交互的主要Python接口
    # 它需要一个节点名来在后台创建自己的节点用于通信
    logger.info("初始化 MoveItPy...")
    moveit = MoveItPy(node_name="moveit_py")

    # -------------------------------------------------------------------
    # 2. 获取规划组件和当前的机器人状态
    # -------------------------------------------------------------------
    # 'arm_manipulator' 是在SRDF文件中定义的规划组(planning group)的名称。
    # 你需要确认你的机器人配置中这个组的名称是什么。
    # 常见的名称有 'manipulator', 'arm', 'arm_manipulator' 等。
    # **如何查找?**: 在RViz的MotionPlanning插件中，会有一个“Planning Group”下拉菜单，里面的选项就是可用的组名。
    arm = moveit.get_planning_component("arm_manipulator")
    logger.info("获取规划组件 'arm_manipulator'")
    
    robot_model = moveit.get_robot_model()
    robot_state = moveit.get_current_robot_state()


    # -------------------------------------------------------------------
    # 3. 设置目标姿态 (Pose Goal)
    # -------------------------------------------------------------------
    # 创建一个PoseStamped消息来定义目标位置和姿态
    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "base_link"  # 目标姿态是在哪个坐标系下定义的
    
    # 设置目标位置 (x, y, z)，单位是米
    pose_goal.pose.position.x = 0.3
    pose_goal.pose.position.y = 0.1
    pose_goal.pose.position.z = 0.4

    # 设置目标姿态 (四元数: x, y, z, w)
    # 这里是一个大致朝前的姿态
    pose_goal.pose.orientation.x = 0.924
    pose_goal.pose.orientation.y = -0.383
    pose_goal.pose.orientation.z = 0.0
    pose_goal.pose.orientation.w = 0.0

    # 将目标姿态设置到机器人状态对象中
    # 注意：这里的 'link_6' 是末端执行器的名称 (end-effector link)
    # 你需要确认你的机器人末端执行器的link叫什么名字
    # **如何查找?**: 在RViz中，展开RobotModel，找到机械臂最末端的那个link。
    end_effector_link = "link_6"
    if not set_goal_pose(robot_state=robot_state, pose=pose_goal.pose, link_name=end_effector_link):
        logger.error("设置目标姿态失败")
        rclpy.shutdown()
        return

    # -------------------------------------------------------------------
    # 4. 规划运动
    # -------------------------------------------------------------------
    logger.info("开始规划运动...")
    # 使用OMPL作为规划器，这是在你的launch文件中配置的
    planning_params = MultiPipelinePlanRequestParameters(
        moveit, ["ompl_rrtc"]
    )
    plan_result = arm.plan(planning_params=planning_params, single_plan_parameters=None)

    # -------------------------------------------------------------------
    # 5. 执行运动
    # -------------------------------------------------------------------
    if plan_result:
        logger.info("规划成功，准备执行...")
        robot_trajectory = plan_result.trajectory
        moveit.execute(robot_trajectory, controllers=[])
        logger.info("执行完毕!")
    else:
        logger.error("规划失败!")

    # 最终关闭rclpy
    rclpy.shutdown()

def set_goal_pose(robot_state: RobotState, pose: PoseStamped, link_name: str) -> bool:
    """辅助函数，用于设置目标姿态。"""
    robot_state.set_from_ik(
        group_name="arm_manipulator",
        pose_stamped=pose,
        link_name=link_name,
        timeout=3.0,
    )
    return True


if __name__ == "__main__":
    main()