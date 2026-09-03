#include "FeatureExtractor/SuperPointExtractor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace sfm {

namespace {

/**
 * @brief Performs mutual best-match between two sets of L2-normalized
 *        descriptors using cosine similarity, mirroring the official XFeat
 *        match() function (also appropriate for SuperPoint's unit-length
 *        descriptors).
 */
void CrossMatchingSuperPoint(const cv::Mat& desc1, const cv::Mat& desc2,
    std::vector<cv::DMatch>& match_ij_out, double minCossim) {
    match_ij_out.clear();
    if (desc1.rows == 0 || desc2.rows == 0) {
        return;
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    std::vector<cv::DMatch> match12, match21;
    matcher->match(desc1, desc2, match12);
    matcher->match(desc2, desc1, match21);

    const double minDist = std::sqrt(std::max(0.0, 2.0 * (1.0 - minCossim)));
    match_ij_out.reserve(match12.size());
    for (const auto& m : match12) {
        const cv::DMatch& back = match21[m.trainIdx];
        if (back.trainIdx == m.queryIdx && m.distance <= minDist) {
            cv::DMatch mm = m;
            mm.distance = 1.0 - 0.5 * m.distance * m.distance;
            match_ij_out.push_back(mm);
        }
    }
}

}  // namespace

SuperPointExtractor::SuperPointExtractor(const std::string& configFile) {
    std::string weightPath = "superpoint.onnx";
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    if (fs.isOpened()) {
        topK_ = fs["feature"]["superpoint"]["superpoint_top_k"].empty() ? 1024 : (int)fs["feature"]["superpoint"]["superpoint_top_k"];
        nmsRadius_ = fs["feature"]["superpoint"]["superpoint_nms_radius"].empty() ? 4 : (int)fs["feature"]["superpoint"]["superpoint_nms_radius"];
        detectionThreshold_ = fs["feature"]["superpoint"]["superpoint_detection_threshold"].empty() ? 0.015 : (double)fs["feature"]["superpoint"]["superpoint_detection_threshold"];
        matchThreshold_ = fs["feature"]["superpoint"]["superpoint_match_threshold"].empty() ? 0.9 : (double)fs["feature"]["superpoint"]["superpoint_match_threshold"];
        maxDim_ = fs["feature"]["superpoint"]["superpoint_max_dim"].empty() ? 640 : (int)fs["feature"]["superpoint"]["superpoint_max_dim"];
        weightPath = fs["feature"]["superpoint"]["superpoint_weight"].empty() ? std::string("superpoint.onnx") : (std::string)fs["feature"]["superpoint"]["superpoint_weight"];
        fs.release();
    }

    superpoint_ = std::make_shared<SuperPoint>();
    if (!superpoint_->Load(weightPath)) {
        std::cerr << "SuperPointExtractor: failed to load SuperPoint weights: " << weightPath << std::endl;
        superpoint_.reset();
    }
}

bool SuperPointExtractor::Extract(const cv::Mat& image,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors,
                                  std::vector<uint64_t>& binaryDescriptors,
                                  std::vector<float>& scores) {
    std::cout << "use superpoint\n";
    binaryDescriptors.clear();
    if (!superpoint_) {
        return false;
    }
    return superpoint_->detectAndCompute(image, topK_, keypoints, descriptors, scores,
                                         nmsRadius_, detectionThreshold_, maxDim_);
}

bool SuperPointExtractor::ExtractLowRes(const cv::Mat& image,
                                        std::vector<cv::KeyPoint>& keypoints,
                                        cv::Mat& descriptors) {
    if (!superpoint_) {
        return false;
    }
    std::vector<float> scores;
    return superpoint_->detectAndCompute(image, 300, keypoints, descriptors, scores,
                                         nmsRadius_, detectionThreshold_, maxDim_);
}

void SuperPointExtractor::Match(const cv::Mat& desc1, const std::vector<uint64_t>& binary1,
                                const cv::Mat& desc2, const std::vector<uint64_t>& binary2,
                                std::vector<cv::DMatch>& matches) const {
    CrossMatchingSuperPoint(desc1, desc2, matches, matchThreshold_);
}

void SuperPointExtractor::MatchLowRes(const cv::Mat& desc1, const cv::Mat& desc2,
                                      std::vector<cv::DMatch>& matches) const {
    CrossMatchingSuperPoint(desc1, desc2, matches, matchThreshold_);
}

}  // namespace sfm
