import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
from numpy import pi

'''
requirement:
numpy == 1.26.4
'''

def plan_path_and_get_manipulability(robot, T_start, T_end, num_steps=100):
    """
    规划路径，计算IK并获取可操作性指数。
    
    参数:
    robot (rtb.DHRobot): 机器人模型。
    T_start (SE3): 起始位姿。
    T_end (SE3): 结束位姿。
    num_steps (int): 路径点数量。
    
    返回:
    tuple: (路径点列表, 可操作性指数列表, 关节角度列表)
    """
    print("开始路径规划...")
    
    # 1. 生成平滑的笛卡尔轨迹
    # ctraj (Cartesian trajectory) 生成从T_start到T_end的SE3对象列表
    traj = rtb.ctraj(T_start, T_end, num_steps)
    
    path_points = []
    manipulability_indices = []
    joint_path = []
    
    # 2. 求解轨迹上每个点的逆运动学
    q_current = None # 初始猜测为空
    for T in traj:
        # ikine_LM 是一个鲁棒的IK求解器 (Levenberg-Marquardt)
        # 它返回一个包含解、成功状态、迭代次数等的元组
        sol = robot.ikine_LM(T, q0=q_current, ilimit=50)
        
        if sol.success:
            q_solution = sol.q
            
            # 保存路径点 (从SE3对象中提取位置)
            path_points.append(T.t)
            
            # 3. 计算可操作性
            # 工具箱自带了计算可操作性的函数！
            m = robot.manipulability(q_solution, method='yoshikawa')
            manipulability_indices.append(m)
            
            # 保存关节角度
            joint_path.append(q_solution)
            
            # 更新下一次IK的初始猜测值，以保证路径平滑
            q_current = q_solution
        else:
            print(f"Ik failed on the path")

    print("Path planning FINISHED")
    return np.array(path_points), np.array(manipulability_indices), np.array(joint_path)


# --- 主程序 ---
if __name__ == "__main__":
    # 根据图片中的表格定义经典DH参数
    # 同样地，我们将 alpha=410.0 视为笔误并设为0.0。
    
    d1 = -(156.43 + 128.38) / 1000.0 # 假设单位是mm，转换为米
    d2 = -5.38 / 1000.0
    d3 = -6.38 / 1000.0
    d4 = -(208.43 + 105.93) / 1000.0
    d6 = -(105.93 + 61.53) / 1000.0

    # 使用RevoluteDH定义每个连杆。参数顺序为 (d, a, alpha, offset)
    # offset 对应 theta 的常数偏移
    links = [
        rtb.RevoluteDH(d=0, a=0, alpha=pi, offset=0),
        rtb.RevoluteDH(d=d1, a=0, alpha=pi/2, offset=0),
        rtb.RevoluteDH(d=d2, a=0.41, alpha=pi, offset=-np.pi/2),
        rtb.RevoluteDH(d=d3, a=0, alpha=pi/2, offset=-np.pi/2),
        rtb.RevoluteDH(d=d4, a=0, alpha=pi/2, offset=np.pi),
        rtb.RevoluteDH(d=0, a=0, alpha=pi/2, offset=np.pi),
        rtb.RevoluteDH(d=d6, a=0, alpha=pi, offset=np.pi)
    ]
    
    # 从连杆列表创建DHRobot对象
    # 注意：表格有7行(0-6)，这对应一个7自由度的机械臂。
    # 如果关节1-6是可动的，那这是一个6自由度的机械臂，我们应该只用6个连杆。
    # 这里我们假设关节1-6是可动的，第0行是基座变换。
    # DHRobot的base属性可以设置这个固定变换。
    base_transform = SE3(0.2, 0.3, 0.5) * SE3.RPY([0, 0, np.pi], order='xyz')
    robot = rtb.DHRobot(links[1:], name="MyRobot", base=base_transform)
    print(robot) # 打印机器人摘要信息

    # 定义起始和结束位姿
    # 起始位姿：让机器人处于一个初始姿态，例如零角度
    q_start = np.zeros(robot.n) # robot.n 是自由度数量 (6)
    T_start = robot.fkine(q_start) # 计算零角度时的位姿

    # 结束位姿：位置 + 姿态 (使用SE3对象)
    # 位置(x, y, z)和姿态(roll, pitch, yaw)
    T_end = SE3(0.5, 0.3, 0.5) * SE3.RPY([0, 0, np.pi], order='xyz')
    
    # 规划路径并计算
    path, manipulability, joints = plan_path_and_get_manipulability(robot, T_start, T_end, num_steps=100)
    
    # --- 可视化 ---
    if path.shape[0] > 0:
        fig = plt.figure(figsize=(15, 6))
        
        # 1. 可视化笛卡尔路径
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'b-o', markersize=3, label='End-effector Path')
        ax1.scatter(path[0, 0], path[0, 1], path[0, 2], c='green', s=100, label='Start', zorder=10)
        ax1.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='red', s=100, label='End', zorder=10)
        ax1.set_title('Cartesian Path of the End-Effector')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.grid(True)
        # 保持坐标轴比例一致
        ax1.set_aspect('equal')

        # 2. 可视化可操作性指数
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(manipulability, 'r-')
        ax2.set_title('Manipulability Index Along the Path')
        ax2.set_xlabel('Path Step')
        ax2.set_ylabel('Manipulability Index (Yoshikawa)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()