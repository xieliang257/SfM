#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sfm {

/**
 * @brief Abstract interface for a feature extraction/matching backend.
 *
 * Each backend owns everything that is specific to a particular feature type
 * (SIFT, XFeat, SuperPoint, ...): keypoint detection, descriptor computation,
 * and the matching strategy used to relate two images. The SfM pipeline
 * (Frame / ImageProcessor) only talks to this interface and stays agnostic
 * about which feature backend is configured.
 *
 * The matching entry points come in two flavours:
 *  - Match():      full-resolution matching, used to produce the final
 *                  inter-frame correspondences.
 *  - MatchLowRes(): matching on the low-resolution descriptors, used to build
 *                  the coarse match graph that prunes expensive full matches.
 *
 * Some backends (e.g. SIFT) additionally produce binary descriptors that speed
 * up full-resolution matching. These are stored per frame and handed back to
 * Match() so the backend can decide whether to use them.
 */
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    /// Human-readable name of the feature backend (e.g. "sift", "xfeat").
    virtual std::string Name() const = 0;

    /// Optional extra detail about the inference mode (e.g. "hand-written" /
    /// "onnx") shown in the debug overlay. Empty means no extra detail.
    virtual std::string EngineMode() const { return ""; }

    /// Extract full-resolution features. Returns false when no keypoints
    /// could be produced.
    virtual bool Extract(const cv::Mat& image,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors,
                         std::vector<uint64_t>& binaryDescriptors,
                         std::vector<float>& scores) = 0;

    /// Extract low-resolution features (used only for the match graph).
    virtual bool ExtractLowRes(const cv::Mat& image,
                               std::vector<cv::KeyPoint>& keypoints,
                               cv::Mat& descriptors) = 0;

    /// Match two full-resolution feature sets. binary1/binary2 may be empty
    /// for backends that do not produce binary descriptors.
    virtual void Match(const cv::Mat& desc1, const std::vector<uint64_t>& binary1,
                       const cv::Mat& desc2, const std::vector<uint64_t>& binary2,
                       std::vector<cv::DMatch>& matches) const = 0;

    /// Match two low-resolution feature sets.
    virtual void MatchLowRes(const cv::Mat& desc1, const cv::Mat& desc2,
                             std::vector<cv::DMatch>& matches) const = 0;
};

}  // namespace sfm
