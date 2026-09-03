#pragma once
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor/FeatureExtractor.h"
#include "XFeat/XFeat.h"

namespace sfm {

/**
 * @brief FeatureExtractor backend built on the self-contained C++ XFeat
 *        engine. Owns the shared XFeat instance (weights are loaded once and
 *        reused for every frame) and the XFeat-specific matching with a
 *        cosine-similarity threshold.
 */
class XFeatExtractor : public FeatureExtractor {
public:
    explicit XFeatExtractor(const std::string& configFile);

    std::string Name() const override { return "xfeat"; }

    std::string EngineMode() const override {
        return (xfeat_ && xfeat_->UsesONNX()) ? "onnx" : "hand-written";
    }

    bool Extract(const cv::Mat& image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors,
                 std::vector<uint64_t>& binaryDescriptors,
                 std::vector<float>& scores) override;

    bool ExtractLowRes(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors) override;

    void Match(const cv::Mat& desc1, const std::vector<uint64_t>& binary1,
               const cv::Mat& desc2, const std::vector<uint64_t>& binary2,
               std::vector<cv::DMatch>& matches) const override;

    void MatchLowRes(const cv::Mat& desc1, const cv::Mat& desc2,
                     std::vector<cv::DMatch>& matches) const override;

private:
    std::shared_ptr<XFeat> xfeat_;
    int topK_ = 2000;
    double matchThreshold_ = 0.82;
};

}  // namespace sfm
