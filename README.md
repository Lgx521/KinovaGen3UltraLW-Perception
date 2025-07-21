# KinovaGen3UltraLW-Perception
### This repo is for the develpment of a kinova gen3 ultra light-weight robot arm.
---
## Modules
#### kinova_camera_streamer
- For streaming the videos from the build in camera
- Usage: `ros2 launch kinova_camera_streamer start_camera_with_tf.launch.py`
- Techiques: gstreamer
- Publishing topic: `/kinova_camera/color/image_raw`
- 
#### arm_action_controller
- For control by sending a action message
- Example:
  - Close the gripper (1.0--closed, 0.0--opened)
    ```bash 
    ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command:{position: 1.0, max_effort: 100.0}}"
    ```

  - Move to specific postion (pos+quarternion)
    Parallel to the XY plan (base link)
    ```bash 
    ros2 action send_goal /move_to_pose arm_action_controller/action/MoveToPose "{
      target_pose: {
        header: { frame_id: 'base_link' },
        pose: {
          position: { x: 0.3, y: 0.0, z: 0.45 },
          orientation: { x: 0.0, y: 0.707, z: 0.0, w: 0.707 }
        }
      }
    }" --feedback 
    ```
    Head down the end effector (Parallel to -Z direction)
    ```bash 
    ros2 action send_goal /move_to_pose arm_action_controller/action/MoveToPose "{
      target_pose: {
        header: { frame_id: 'base_link' },
        pose: {
          position: { x: 0.3, y: 0.0, z: 0.45 },
          orientation: { x: 0.0, y: 1, z: 0.0, w: 0.0 }
        }
      }
    }" --feedback 
    ```

## Testing
#### XYZ static angle converting to Quarternion
Use the script `XYZ_to_quarternion.py` @ `/scripts/XYZ_to_quarternion.py` 


    ros2 action send_goal /move_to_pose arm_action_controller/action/MoveToPose "{
      target_pose: {
        header: { frame_id: 'base_link' },
        pose: {
          position: { x: 0.3866, y: 0.0694, z: 0.032 },
          orientation: { x: -0.009629, y: -0.000665, z: -0.999802, w: 0.017419 }
        }
      }
    }" --feedback 


   position: { x: 0.3867, y: 0.0694, z: 0.0262 },
  orientation: { x: -0.009629, y: -0.000665, z: -0.999802, w: 0.017419 }

  [INFO] [1753123606.496630089] [apriltag_detector_node]: --- Pose of Marker ID: 0 in 'base_link' frame ---
  position: { x: 0.3813, y: 0.1117, z: 0.0256 },
  orientation: { x: 0.008357, y: 0.001405, z: 0.999437, w: -0.032456 }
  orientation_ang: { rx: -0.0257, ry: -0.0043, rz: -3.0766 }