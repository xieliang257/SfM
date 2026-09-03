#pragma once
#include <opencv2/opencv.hpp>
#include "FeatureExtractor/FeatureExtractor.h"

namespace sfm {

/**
 * @brief FeatureExtractor backend built on OpenCV SIFT.
 *
 * Replicates the original SIFT pipeline: uniform block-based keypoint
 * detection, float descriptors normalized to unit length, binary descriptors
 * derived from them, and hybrid binary/float brute-force matching.
 */
class SiftExtractor : public FeatureExtractor {
public:
    explicit SiftExtractor(const std::string& configFile);

    std::string Name() const override { return "sift"; }

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
    int featureCount_ = 12000;
    cv::Ptr<cv::SIFT> sift_;
};

}  // namespace sfm
