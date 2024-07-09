# Structure from Motion (SfM)

## System Overview

This project is an SfM (Structure from Motion) system that utilizes computer vision and image processing technologies to 
reconstruct three-dimensional structures from multiple photographs. Aiming to provide developers with a complete, basic SfM solution.


## Development Environment

- **Operating System**: Windows 10 or higher
- **Development Environment**: Visual Studio 2017 or higher
- **Main Dependencies**:
  - **Ceres Solver**: Used for Bundle Adjustment (BA) optimization
  - **OpenCV 4.4.0 or higher**: 
      Used for image processing and computer vision operations, especially as newer versions include the SIFT algorithm in the main module, 
	  eliminating the need for the extra contrib module.
  - **Eigen**: Provides efficient matrix and vector calculations.

## Installation Guide

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
	
3. **Install Eigen via vcpkg**:
	vcpkg install eigen3:x64-windows

4. **OpenCV**:
- We recommend downloading the precompiled binaries of OpenCV directly from (https://opencv.org/releases/).
  This approach simplifies the installation process, especially for users who may not be familiar with building from source.
	
## Building the Project

The project is built using CMake. For ease of configuration, it's recommended to use CMake GUI.

- **Download CMake GUI**: You can download CMake GUI from (https://cmake.org/download/).