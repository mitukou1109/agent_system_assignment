cmake_minimum_required(VERSION 3.8)
project(vnoid_challenge)

find_package(ament_cmake_auto REQUIRED)
find_package(Torch REQUIRED)

ament_auto_find_build_dependencies()

choreonoid_add_simple_controller(Go2Controller src/Go2Controller.cpp)
target_link_libraries(Go2Controller
  ${CHOREONOID_BASE_LIBRARIES}
  ${TORCH_LIBRARIES}
)

ament_auto_package(
  INSTALL_TO_SHARE
    model
    project
    scripts
)
