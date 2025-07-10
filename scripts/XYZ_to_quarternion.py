import numpy as np
from scipy.spatial.transform import Rotation as R

def xyz_to_quaternion(roll, pitch, yaw, degrees=True):
  """
  将XYZ固定角（Tait-Bryan角）转换为四元数。
  旋转顺序为x, y, z。

  参数:
      roll (float): 绕x轴的旋转角度。
      pitch (float): 绕y轴的旋转角度。
      yaw (float): 绕z轴的旋转角度。
      degrees (bool): 输入的角度是否为度数。默认为True。

  返回:
      numpy.ndarray: 表示旋转的四元数 (x, y, z, w)。
  """
  # from_euler 函数的 'xyz' 参数指定了旋转顺序
  # degrees=True 表示输入的是角度值
  r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
  
  # as_quat() 返回一个 (x, y, z, w) 形式的四元数
  return r.as_quat()

# --- 示例 ---
if __name__ == '__main__':
  # 定义XYZ固定角（单位：度）
  roll_angle = 0.0
  pitch_angle = 180.0
  yaw_angle = 0.0

  # 进行转换
  quaternion = xyz_to_quaternion(roll_angle, pitch_angle, yaw_angle)

  # 打印结果
  print(f"XYZ 固定角 (roll, pitch, yaw): ({roll_angle}, {pitch_angle}, {yaw_angle}) 度")
  # scripy返回的四元数格式为 (x, y, z, w)
  print(f"转换后的四元数 (x, y, z, w): {quaternion}")

  # 你也可以选择以弧度作为输入
  # roll_rad = np.deg2rad(roll_angle)
  # pitch_rad = np.deg2rad(pitch_angle)
  # yaw_rad = np.deg2rad(yaw_angle)
  # quaternion_rad = xyz_to_quaternion(roll_rad, pitch_rad, yaw_rad, degrees=False)
  # print(f"\n以弧度作为输入的转换结果: {quaternion_rad}")
