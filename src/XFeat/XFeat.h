#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"

namespace sfm {

/**
 * @brief A self-contained C++ implementation of the XFeat feature extractor
 *        (Potje et al., CVPR 2024: "XFeat: Accelerated Features for Lightweight
 *        Image Matching").
 *
 * The convolutional backbone is executed either by a hand-written CNN engine
 * (no external deep-learning runtime is required) using weights exported from
 * the official PyTorch checkpoint with tools/export_xfeat_weights.py, or by an
 * ONNX Runtime session over the model produced by tools/export_xfeat_onnx.py
 * (weights/xfeat.onnx), which is numerically identical to the hand-written
 * forward pass. The sparse keypoint post-processing (softmax heatmap, NMS,
 * top-k, descriptor sampling) is implemented here as well, so the whole
 * pipeline runs in C++ on arbitrary image sizes.
 */
class XFeat {
public:
    XFeat();
    ~XFeat();

    // Deleted copy to avoid duplicating large weight buffers.
    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;

    /**
     * @brief Loads the network weights from the binary file produced by
     *        tools/export_xfeat_weights.py.
     * @param weightPath Path to the .bin weight file.
     * @return True on success.
     */
    bool Load(const std::string& weightPath);

    /**
     * @brief Loads the ONNX model produced by tools/export_xfeat_onnx.py and
     *        creates an inference session. When a session is loaded it is used
     *        instead of the hand-written Forward(); the hand-written engine is
     *        still kept in the binary and used when no session is present.
     * @param modelPath Path to the .onnx model file.
     * @return True on success.
     */
    bool LoadONNX(const std::string& modelPath);

    /// @brief Whether the ONNX Runtime session is active (used for display).
    bool UsesONNX() const { return onnxLoaded_; }

    /**
     * @brief Extracts sparse keypoints and their descriptors from an image.
     * @param gray Input grayscale image (CV_8U or CV_32F in [0, 1], any size).
     * @param topK Maximum number of keypoints to return.
     * @param kpts Output keypoints (full-resolution image coordinates).
     * @param descriptors Output CV_32F matrix, one 64-D L2-normalized row per keypoint.
     * @param scores Output reliability score per keypoint.
     * @return True if at least one keypoint was produced.
     */
    bool detectAndCompute(const cv::Mat& gray, int topK,
                          std::vector<cv::KeyPoint>& kpts,
                          cv::Mat& descriptors,
                          std::vector<float>& scores);

private:
    struct Tensor4 {
        int C = 0, H = 0, W = 0;
        std::vector<float> d;

        void Resize(int c, int h, int w) {
            C = c; H = h; W = w;
            d.assign(size_t(c) * h * w, 0.f);
        }
        float* Ptr(int c, int y, int x) { return d.data() + (size_t(c) * H + y) * W + x; }
        const float* Ptr(int c, int y, int x) const { return d.data() + (size_t(c) * H + y) * W + x; }
    };

    // ---- CNN layers -----------------------------------------------------
    void Conv(const Tensor4& in, const float* weight, const float* bias,
              bool hasBias, int outC, int k, int stride, int pad, Tensor4& out);
    void Conv1x1(const Tensor4& in, const float* weight, const float* bias,
                 bool hasBias, int outC, Tensor4& out);
    void BatchNorm(const Tensor4& in, const float* scale, const float* shift, Tensor4& out);
    void Relu(Tensor4& x);
    void InstanceNorm1(Tensor4& x);
    void AvgPool4(const Tensor4& in, Tensor4& out);
    void ResizeBilinear(const Tensor4& in, int outH, int outW, Tensor4& out);
    void Add3(const Tensor4& a, const Tensor4& b, const Tensor4& c, Tensor4& out);
    void Unfold8(const Tensor4& in, Tensor4& out);

    // ---- Network --------------------------------------------------------
    void Forward(const cv::Mat& gray, int H, int W,
                 Tensor4& feats, Tensor4& keypoints, Tensor4& heatmap);
    // ONNX Runtime equivalent of Forward(); fills the same three tensors.
    bool ForwardONNX(const cv::Mat& gray, int H, int W,
                     Tensor4& feats, Tensor4& keypoints, Tensor4& heatmap);
    void ortCheck(OrtStatus* s);

    // ---- Sparse post-processing ----------------------------------------
    void BuildHeatmap(const Tensor4& keypoints, cv::Mat& heatmap);
    void NMS(const cv::Mat& heatmap, float threshold, int kernel, std::vector<cv::Point2i>& pts);
    void SampleBilinear(const Tensor4& coarse, const std::vector<cv::Point2i>& fullPts,
                        int fullH, int fullW, std::vector<float>& vals);
    void SampleBicubic(const Tensor4& coarse, const std::vector<cv::Point2i>& fullPts,
                       int fullH, int fullW, cv::Mat& descriptors);

    // ---- Weight store ---------------------------------------------------
    bool loaded_ = false;
    std::vector<float> weights_;    // Layer weights (pointers into weights_).
    const float* w_skip1 = nullptr;  const float* b_skip1 = nullptr;
    const float* w_b1[4] = {}; const float* s_b1[4] = {}; const float* h_b1[4] = {};
    const float* w_b2[2] = {}; const float* s_b2[2] = {}; const float* h_b2[2] = {};
    const float* w_b3[3] = {}; const float* s_b3[3] = {}; const float* h_b3[3] = {};
    const float* w_b4[3] = {}; const float* s_b4[3] = {}; const float* h_b4[3] = {};
    const float* w_b5[4] = {}; const float* s_b5[4] = {}; const float* h_b5[4] = {};
    const float* w_bf[3] = {}; const float* s_bf[2] = {}; const float* h_bf[2] = {};
    const float* b_bf2 = nullptr;
    const float* w_hh[3] = {}; const float* s_hh[2] = {}; const float* h_hh[2] = {};
    const float* b_hh2 = nullptr;
    const float* w_kh[4] = {}; const float* s_kh[3] = {}; const float* h_kh[3] = {};
    const float* b_kh3 = nullptr;

    // ---- ONNX Runtime session (optional) -------------------------------
    bool onnxLoaded_ = false;
    const OrtApi* ort = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memInfo = nullptr;
    OrtAllocator* allocator = nullptr;
    char* inputName = nullptr;
    char* featsName = nullptr;
    char* keypointsName = nullptr;
    char* heatmapName = nullptr;
};

}  // namespace sfm
