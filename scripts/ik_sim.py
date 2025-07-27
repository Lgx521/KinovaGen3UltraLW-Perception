import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
from numpy import pi

'''
requirement:
numpy <= 2.0.0
'''

def plot_frame(ax, T, length=0.005):
    """
    在给定的3D坐标轴上绘制一个坐标系。
    
    参数:
    ax (Axes3D): Matplotlib的3D坐标轴对象。
    T (SE3): 要可视化的位姿。
    length (float): 坐标轴的显示长度。
    """
    origin = T.t
    x_axis = T.R[:, 0]  # X轴向量 (旋转矩阵的第一列)
    y_axis = T.R[:, 1]  # Y轴向量 (旋转矩阵的第二列)
    z_axis = T.R[:, 2]  # Z轴向量 (旋转矩阵的第三列)
    
    # 绘制X轴 (红色)
    ax.plot([origin[0], origin[0] + length * x_axis[0]], 
            [origin[1], origin[1] + length * x_axis[1]], 
            [origin[2], origin[2] + length * x_axis[2]], color='red', linewidth=2)
    
    # 绘制Y轴 (绿色)
    ax.plot([origin[0], origin[0] + length * y_axis[0]], 
            [origin[1], origin[1] + length * y_axis[1]], 
            [origin[2], origin[2] + length * y_axis[2]], color='green', linewidth=2)
    
    # 绘制Z轴 (蓝色)
    ax.plot([origin[0], origin[0] + length * z_axis[0]], 
            [origin[1], origin[1] + length * z_axis[1]], 
            [origin[2], origin[2] + length * z_axis[2]], color='blue', linewidth=2)


def plan_path_and_get_manipulability(robot, q_start, T_end, num_steps=100):
    """
    规划路径，计算IK并获取可操作性指数。
    
    返回:
    tuple: (完整位姿轨迹, 可操作性指数列表, 关节角度列表)
    """
    print("开始路径规划...")
    
    T_start = robot.fkine(q_start)
    traj = rtb.ctraj(T_start, T_end, num_steps)
    
    successful_traj = [] # 存储成功的SE3位姿
    manipulability_indices = []
    joint_path = []
    
    q_current = q_start
    
    for T in traj:
        sol = robot.ikine_LM(T, q0=q_current, ilimit=30, joint_limits=True)
        
        if sol.success:
            q_solution = sol.q
            successful_traj.append(T) # 保存完整的SE3对象
            
            m = robot.manipulability(q_solution, method='yoshikawa')
            manipulability_indices.append(m)
            
            joint_path.append(q_solution)
            q_current = q_solution
        else:
            print(f"警告: 逆运动学在轨迹的某一点求解失败。")
            break

    print(f"路径规划完成。成功规划了 {len(successful_traj)}/{num_steps} 个点。")
    return successful_traj, np.array(manipulability_indices), np.array(joint_path)


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
    
    robot = rtb.DHRobot(links, name="MyDebugRobot")
    print("--- 用于调试的机器人模型 ---")
    print(robot)
    
    q_start = np.zeros(robot.n)
    T_start = SE3(0.5, 0.3, 0.5) * SE3.RPY([0, 0, np.pi], order='xyz')

    # 创建一个可达的目标位姿
    T_end = SE3(-0.5, 0.3, 0.5) * SE3.RPY([0, 0, 0], order='xyz')
    print(f"\n设定一个可达的目标位姿: \n{T_end}")
    
    # 规划路径并计算
    full_traj, manipulability, joints = plan_path_and_get_manipulability(robot, q_start, T_end, num_steps=100)
    
    # --- 可视化 ---
    if joints.shape[0] > 0:
        # 可视化机器人动画 (可选，会生成一个GIF)
        # print("正在生成机器人运动动画GIF...")
        # robot.plot(joints, movie='robot_path.gif')

        fig = plt.figure(figsize=(15, 6))
        
        # 1. 可视化笛卡尔路径和位姿
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # 从SE3对象列表中提取位置点用于绘制路径线
        path_points = np.array([T.t for T in full_traj])
        ax1.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'c--', label='End-effector Path')
        
        # --- 新增：绘制位姿坐标系 ---
        plot_interval = 5  # 每隔10个点绘制一次坐标系，避免图形过于杂乱
        axis_length = 0.02  # 坐标轴的显示长度
        
        # 绘制路径上间隔的位姿
        for T in full_traj[::plot_interval]:
            plot_frame(ax1, T, length=axis_length)

        # 绘制起点和终点的位姿
        # for T in full_traj[0:-1:len(full_traj)-1]:
        #     plot_frame(ax1, T, length=axis_length)

        # 确保绘制起始和结束位姿
        plot_frame(ax1, full_traj[0], length=axis_length)
        plot_frame(ax1, full_traj[-1], length=axis_length)

        ax1.set_title('Cartesian Path and Pose of the End-Effector')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.grid(True)
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
