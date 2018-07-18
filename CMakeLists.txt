cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(segDemoCPP)

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")

set(CMAKE_BUILD_TYPE "Debug")
# cuda9.0
INCLUDE_DIRECTORIES(/usr/local/cuda-9.0/include)
LINK_DIRECTORIES(/usr/local/cuda-9.0/lib64)
# OpenCV33
INCLUDE_DIRECTORIES(/usr/local/include)
LINK_DIRECTORIES(/usr/local/lib)
# caffe
INCLUDE_DIRECTORIES(~/caffe/include)
LINK_DIRECTORIES(~/caffe/build/lib)
# boost, gflags, glog
INCLUDE_DIRECTORIES(/usr/include)
LINK_DIRECTORIES(/usr/lib)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

#SET(TEST_ONE ${DIR_SRCS})

set(OpenCV_LIBRARIES opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)

ADD_EXECUTABLE(${PROJECT_NAME} src/caffe_cpp_demo.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} caffe glog gflags boost_system ${OpenCV_LIBRARIES})
# cudart cuda