cmake_minimum_required(VERSION 3.8)
project(arm_action_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find necessary packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# This is the key command to generate Python code from your .action file
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveToPose.action"
  DEPENDENCIES geometry_msgs action_msgs
)

# Create a directory for our python scripts
file(GLOB_RECURSE python_scripts "src/*.py")
install(PROGRAMS ${python_scripts}
  DESTINATION lib/${PROJECT_NAME}
)

# Standard ament_python setup for entry points
# ament_python_install_package(${PROJECT_NAME})

# Install launch files and config files if you have them
install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()