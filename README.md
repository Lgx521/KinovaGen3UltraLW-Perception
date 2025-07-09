# KinovaGen3UltraLW-Perception
### This repo is for the develpment of a kinova gen3 ultra light-weight robot arm.
---
## Modules
- kinova_camera_streamer
  - For streaming the videos from the build in camera
  - Usage: `ros2 launch kinova_camera_streamer start_camera_with_tf.launch.py`
  - Techiques: gstreamer
  - Publishing topic: `/kinova_camera/color/image_raw`