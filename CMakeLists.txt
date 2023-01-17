cmake_minimum_required(VERSION 3.23)
project(FourPoint_tenorrtv7)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA)#加这一句定位cuda包
find_package(OpenCV REQUIRED) #加这一句
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

#这两句是你tensorrt所在的位置
set(TENSORRT_PATH /home/yuesang/include/TensorRT-8.4.1.5/)
set(TENSORRT_LIB_PATH /home/yuesang/include/TensorRT-8.4.1.5/lib)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}) #加这一句
include_directories(${TENSORRT_PATH}/include) #加这一句
include_directories(${CUDA_INCLUDE_DIRS}) #加这一句
include_directories(${Fmt_INCLUDE_DIRS}) #加这一句

set(SOURCE_FILES
        main.cpp
        AutoShoot/Logger/Logger.h
        AutoShoot/TRT/TRTModule.h
        AutoShoot/TRT/TRTModule.cpp
        )
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")#添加这句

cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES} cuda fmt ${Eigen3_LIBRARIES})#加这一句及