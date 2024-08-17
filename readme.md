# Structure from Motion (SfM)

## System Overview

This project is an SfM (Structure from Motion) system that utilizes computer vision and image processing technologies to 
reconstruct three-dimensional structures from multiple photographs. Aiming to provide developers with a complete, basic SfM solution.


## Development Environment

- **Operating System**: Windows 10 or higher, Linux
- **Development Environment**: Visual Studio 2017 or higher (Windows), GCC/G++ (Linux)
- **Main Dependencies**:
  - **Ceres Solver**: Used for Bundle Adjustment (BA) optimization
  - **OpenCV 4.4.0 or higher**: 
      Used for image processing and computer vision operations, especially as newer versions include the SIFT algorithm in the main module, 
	  eliminating the need for the extra contrib module.
  - **Eigen**: Provides efficient matrix and vector calculations.

## Installation Guide

Windows Installation

It is recommended to use [vcpkg](https://github.com/microsoft/vcpkg) to manage and install project dependencies to simplify the configuration process.

1. **Install vcpkg** (if not already installed):
  ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.bat
   ./vcpkg integrate install
   
2. **Install Ceres Solver via vcpkg**:
  ```bash
  vcpkg install ceres:x64-windows

4. **OpenCV**:
- We recommend downloading the precompiled binaries of OpenCV directly from (https://opencv.org/releases/).
  This approach simplifies the installation process, especially for users who may not be familiar with building from source.
  
Linux Installation

1. **Install Denpendencies**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake build-essential libopencv-dev libeigen3-dev
	
2. **Install Ceres Solver:**:
  ```bash
  sudo apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
  git clone https://ceres-solver.googlesource.com/ceres-solver
  mkdir ceres-bin
  cd ceres-bin
  cmake ../ceres-solver
  make -j4
  sudo make install
	
3. **Install OpenCV 4.4.0 or higher**:
  ```bash
  sudo apt-get install -y libopencv-dev
  sudo apt-get remove -y libopencv-dev
  sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
  sudo apt-get install -y python3.8-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

  git clone https://github.com/opencv/opencv.git
  cd opencv
  git checkout 4.4.0
  mkdir build
  cd build
  cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
  make -j4
  sudo make install
	
## Building the Project

Windows:
The project is built using CMake. For ease of configuration, it's recommended to use CMake GUI.

- **Download CMake GUI**: You can download CMake GUI from (https://cmake.org/download/).

Linux:
  mkdir build
  cd build
  cmake ..
  make

Note:
  If the camera is uncalibrated, please set the focal length to -1 in the configuration parameters. The initial focal length will be automatically calculate. 
  If optimization of internal parameters is specified, the focal length and distortion coefficients will also be optimized during the bundle adjustment (BA) process.