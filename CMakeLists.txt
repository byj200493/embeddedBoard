cmake_minimum_required(VERSION 3.14.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O3 -std=c++14 -fopenmp")

project(three_camControl)

find_package(OpenCV 4 REQUIRED)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(
  ./include
  /usr/local/include
  ${OpenCV_INCLUDE_DIRS}
  )



LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

#"option"用来定义宏，"ON"表示打开，"OFF"表示关闭
option (USE_TENSORRT "use tensorrt yolov4" OFF)
IF(USE_TENSORRT)
  add_definitions("-DUSE_TENSORRT")
endif(USE_TENSORRT)

if(USE_TENSORRT)
  add_definitions(-std=c++14)
  option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_BUILD_TYPE Debug)
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++14;-g;-G;-gencode;arch=compute_30;code=sm_30)
  include_directories(${PROJECT_SOURCE_DIR}/include)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      message("embed_platform on")
      include_directories(/usr/local/cuda/targets/aarch64-linux/include)
      link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
  else()
      message("embed_platform off")
      include_directories(/usr/local/cuda/include)
      link_directories(/usr/local/cuda/lib64)
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
  #cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/tensorrt_yolov4/yololayer.cu ${PROJECT_SOURCE_DIR}/src/tensorrt_yolov4/mish.cu)
  #include_directories(./include/tensorrt_yolov4)

  add_executable(three_camControl main.cpp src/camControl.cpp src/render.cpp src/utility.cpp)

else()

  add_executable(three_camControl main.cpp src/camControl.cpp src/render.cpp src/utility.cpp)

endif(USE_TENSORRT)

target_link_libraries(three_camControl 
-lsniris 
-ljsoncpp
${OpenCV_LIBS} 
${PCL_LIBRARIES}
-lcrypto
-lssl
-lcpprest
-pthread
-lzmq)

if(USE_TENSORRT)
  target_link_libraries(three_camControl myplugins)
  target_link_libraries(three_camControl nvinfer)
  target_link_libraries(three_camControl cudart)
endif(USE_TENSORRT)