cmake_minimum_required(VERSION 3.5)

if(WIN32)
set(VCPKG_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Path to vcpkg toolchain file")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_TOOLCHAIN_FILE})
endif()

project(SFMDemo)

find_package(OpenCV 4 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

if(MSVC)
    # Enable AVX2 + FMA so the vectorized descriptor-distance kernels
    # (L2Sqr64/L2Sqr128) compile to the fast SIMD path instead of the scalar
    # fallback, matching the performance seen on the Ubuntu/Linux build.
    add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)
endif()

add_subdirectory(src)
add_subdirectory(samples)
