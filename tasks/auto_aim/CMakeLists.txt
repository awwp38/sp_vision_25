# =====================================================================
# auto_aim CMakeLists.txt
# Supports both OpenVINO (Intel NUC) and TensorRT (Jetson) backends.
#
# For Jetson Orin NX (TensorRT):
#   cmake -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release ..
#
# For Intel NUC (OpenVINO, default):
#   cmake -DCMAKE_BUILD_TYPE=Release ..
# =====================================================================
cmake_minimum_required(VERSION 3.16)

option(USE_TENSORRT "Build with TensorRT backend for Jetson" OFF)

add_subdirectory(planner/tinympc)

set(AUTO_AIM_SOURCES
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

if(USE_TENSORRT)
    message(STATUS "auto_aim: Building with TensorRT backend (Jetson)")
    add_definitions(-DUSE_TENSORRT)
    list(APPEND AUTO_AIM_SOURCES yolos/trt_yolo.cpp)

    # TensorRT & CUDA (provided by JetPack on Jetson)
    find_library(NVINFER_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
    find_library(NVONNXPARSER_LIB nvonnxparser PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)
    find_library(CUDA_LIB cudart PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)
    find_library(CUDNN_LIB cudnn PATHS /usr/lib/aarch64-linux-gnu /usr/local/lib)

    if(NOT NVINFER_LIB)
        message(FATAL_ERROR "TensorRT nvinfer not found! Is JetPack installed?")
    endif()

    add_library(auto_aim OBJECT ${AUTO_AIM_SOURCES})
    target_include_directories(auto_aim PRIVATE
        /usr/include/aarch64-linux-gnu
        /usr/local/cuda/include
    )
    target_link_libraries(auto_aim io tinympcstatic
        ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${CUDA_LIB} ${CUDNN_LIB})

else()
    message(STATUS "auto_aim: Building with OpenVINO backend (Intel)")
    set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    add_library(auto_aim OBJECT ${AUTO_AIM_SOURCES})
    target_link_libraries(auto_aim io openvino::runtime tinympcstatic)
endif()
