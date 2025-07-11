cmake_minimum_required(VERSION 3.8)
project(vnoid_challenge)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

choreonoid_add_simple_controller(InvertedPendulumController
  src/inverted_pendulum_controller.cpp
)
ament_target_dependencies(InvertedPendulumController
  rclcpp
  sensor_msgs
  std_msgs
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/balance_controller.cpp
)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "vnoid_challenge::BalanceController"
  EXECUTABLE balance_controller_node
)

ament_auto_package(
  INSTALL_TO_SHARE
    config
    launch
    model
    project
)
