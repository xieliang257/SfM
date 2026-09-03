#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"

namespace sfm {

/**
 * @brief SuperPoint feature extractor using ONNX Runtime for fast inference.
 *
 * Loads a SuperPoint ONNX model and runs inference via ONNX Runtime C API.
 * The sparse keypoint post-processing (softmax heatmap, subpixel upsampling,
 * fast NMS, border removal, descriptor grid sampling) mirrors the reference
 * SuperPointFrontend implementation.
 */
class SuperPoint {
public:
    SuperPoint();
    ~SuperPoint();

    SuperPoint(const SuperPoint&) = delete;
    SuperPoint& operator=(const SuperPoint&) = delete;

    /**
     * @brief Loads the ONNX model and creates an inference session.
     * @param modelPath Path to the .onnx model file.
     * @return True on success.
     */
    bool Load(const std::string& modelPath);

    /**
     * @brief Extracts sparse keypoints and their descriptors from an image.
     * @param gray Input grayscale image (CV_8U or CV_32F in [0, 1], any size).
     * @param topK Maximum number of keypoints to return.
     * @param kpts Output keypoints (full-resolution image coordinates).
     * @param descriptors Output CV_32F matrix, one 256-D L2-normalized row per keypoint.
     * @param scores Output reliability score per keypoint.
     * @param nmsRadius Non-maximum suppression radius (infinity norm).
     * @param confidenceThreshold Minimum detector confidence to keep a point.
     * @param maxDim Images larger than this (in pixels on the longest side)
     *        are downsampled before running the network.
     * @return True if at least one keypoint was produced.
     */
    bool detectAndCompute(const cv::Mat& gray, int topK,
                          std::vector<cv::KeyPoint>& kpts,
                          cv::Mat& descriptors,
                          std::vector<float>& scores,
                          int nmsRadius = 4,
                          double confidenceThreshold = 0.015,
                          int maxDim = 640);

private:
    // ONNX Runtime objects
    const OrtApi* ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtAllocator* allocator = nullptr;
    char* inputName = nullptr;
    char* semiName = nullptr;
    char* descName = nullptr;

    bool loaded_ = false;

    void ortCheck(OrtStatus* s);
};

}  // namespace sfm
