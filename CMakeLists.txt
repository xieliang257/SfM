cmake_minimum_required(VERSION 3.5)

if(_WIN32)
set(VCPKG_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Path to vcpkg toolchain file")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_TOOLCHAIN_FILE})
endif()

project(SFMDemo)

find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(samples)
