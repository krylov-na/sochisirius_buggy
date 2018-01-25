cmake_minimum_required(VERSION 2.8.3)
project(linedetection)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
  cv_bridge
  image_transport
  tf
)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


catkin_package(
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(linedetection_node src/linedetection_node.cpp src/linedetection.cpp include/linedetection.h)
target_link_libraries(linedetection_node ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(linedetection_node linedetection_generate_messages_cpp)
