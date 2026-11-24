cmake_minimum_required(VERSION 3.16)
project(aim2027 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native")

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui calib3d video)
message(STATUS "OpenCV ${OpenCV_VERSION}")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(aim2027 ${SOURCES})
target_include_directories(aim2027 PRIVATE include)
target_compile_options(aim2027 PRIVATE -Wall -Wextra)
target_link_libraries(aim2027 PRIVATE ${OpenCV_LIBS})
