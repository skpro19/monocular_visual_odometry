cmake_minimum_required(VERSION 2.8)
project(simple_visual_odom)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(
    -DBOOST_COROUTINE_NO_DEPRECATION_WARNING=1
    -DBOOST_ALL_DYN_LINK=1
    -DBOOST_ERROR_CODE_HEADER_ONLY=1
)

add_executable(simple_visual_odom src/visual_odom.cpp src/io.cpp src/main.cpp src/vo_pipeline.cpp)
target_link_libraries(simple_visual_odom
    ${OpenCV_LIBS}
    boost_system
    boost_filesystem
)