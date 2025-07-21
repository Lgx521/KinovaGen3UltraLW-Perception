from scipy.spatial.transform import Rotation as R
            
rotation_z_neg90 = R.from_euler('z', -90, degrees=True).as_matrix()
rotation_y_pos90 = R.from_euler('y', -90, degrees=True).as_matrix()
graspnet_to_tf_rotation = rotation_y_pos90 @ rotation_z_neg90

print(graspnet_to_tf_rotation)