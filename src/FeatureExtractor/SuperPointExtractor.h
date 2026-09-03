#pragma once
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "FeatureExtractor/FeatureExtractor.h"
#include "SuperPoint/SuperPoint.h"

namespace sfm {

/**
 * @brief FeatureExtractor backend built on the self-contained C++ SuperPoint
 *        engine. Owns the shared SuperPoint instance (weights are loaded once
 *        and reused for every frame) and the SuperPoint-specific matching with
 *        a cosine-similarity threshold.
 */
class SuperPointExtractor : public FeatureExtractor {
public:
    explicit SuperPointExtractor(const std::string& configFile);

    std::string Name() const override { return "superpoint"; }

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
    std::shared_ptr<SuperPoint> superpoint_;
    int topK_ = 1024;
    int nmsRadius_ = 4;
    double detectionThreshold_ = 0.015;
    double matchThreshold_ = 0.9;
    int maxDim_ = 640;
};

}  // namespace sfm
