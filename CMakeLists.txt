#
# Root CMakeLists
#
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(Mapper)

#------------------------------------------------------------
# External library dependencies.
#------------------------------------------------------------
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV 4.0.0 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/usr/include/eigen3)
# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include/opencv4)

# add_subdirectory(external_libraries/glfw EXCLUDE_FROM_ALL)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR}/src)
message(STATUS "PROJECT_SOURCE_DIR is " ${PROJECT_SOURCE_DIR})

add_subdirectory(glfw)
add_subdirectory(glm)
include_directories(glm)
add_subdirectory(lib_cg_sandbox)
add_subdirectory(src)
