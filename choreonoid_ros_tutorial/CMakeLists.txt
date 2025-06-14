cmake_minimum_required(VERSION 3.8)
project(choreonoid_ros_tutorial)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

choreonoid_add_simple_controller(RTTTankController src/rtt_tank_controller.cpp)
ament_target_dependencies(RTTTankController
  geometry_msgs
  rclcpp
)

ament_auto_package(
  INSTALL_TO_SHARE
    project
)
