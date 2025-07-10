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
          orientation: { x: 0.0, y: 0.707, z: 0.0, w: 0.707 }
        }
      }
    }" --feedback 
    ```

## Testing
#### XYZ static angle converting to Quarternion
Use the script `XYZ_to_quarternion.py` @ `/scripts/XYZ_to_quarternion.py` 
