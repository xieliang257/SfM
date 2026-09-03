#include "FeatureExtractor/XFeatExtractor.h"
#include "FeatureExtractor/MatchingCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

namespace sfm {

namespace {

/**
 * @brief Converts 64-D L2-normalized float descriptors to a single uint64
 *        binary descriptor (1 bit per dimension, thresholded at 0), mirroring
 *        the SIFT ToBinaryDesc approach. Used to cheaply pre-filter candidate
 *        matches with Hamming distance.
 *
 * XFeat descriptors are L2-normalized around zero (mean ~0), so a per-row
 * mean threshold degenerates to the sign bit anyway; thresholding at 0 is
 * equivalent and slightly better in practice.
 */
void ToBinaryDescXFeat(const cv::Mat& desc, std::vector<uint64_t>& binaryDesc) {
    binaryDesc.resize(desc.rows);
    for (int i = 0; i < desc.rows; ++i) {
        const float* ptr = desc.ptr<float>(i);
        uint64_t b = 0;
        for (int j = 0; j < 64; ++j) {
            b <<= 1;
            if (ptr[j] > 0.f) {
                ++b;
            }
        }
        binaryDesc[i] = b;
    }
}

/**
 * @brief Mixed matching: Hamming distance first narrows the candidates to the
 *        ~20 closest (per descriptor), then the exact L2 cost is computed on
 *        the remaining candidates in float (mirrors the SIFT MixMatching).
 */
void MixMatchingXFeat(const cv::Mat& desc1, const cv::Mat& desc2,
                      const std::vector<uint64_t>& binarys1,
                      const std::vector<uint64_t>& binarys2,
                      const std::vector<bool>& flags,
                      std::vector<cv::DMatch>& matches12) {
    matches12.resize(desc1.rows);
    if (desc1.rows == 0 || desc2.rows == 0) {
        return;
    }

    // Only flagged rows do any work; gathering them up front keeps the parallel
    // partitioning balanced even when the flag set is sparse (reverse pass).
    std::vector<int> active;
    active.reserve(desc1.rows);
    for (int i = 0; i < desc1.rows; ++i) {
        if (flags[i]) {
            active.push_back(i);
        }
    }
    if (active.empty()) {
        return;
    }

    ThreadPool& pool = ThreadPool::Instance();
    const int nWorkers = pool.WorkerCount() + 1;  // +1 for the calling thread

    struct ThreadBuf {
        std::vector<uint32_t> costList;
        std::vector<int> costHist;
        explicit ThreadBuf(int row2) : costList(row2), costHist(65, 0) {}
    };
    std::vector<ThreadBuf> bufs;
    bufs.reserve(nWorkers);
    for (int t = 0; t < nWorkers; ++t) {
        bufs.emplace_back(desc2.rows);
    }

    const int nActive = (int)active.size();
    pool.For(nActive, [&](int r, int tid) {
        const int i = active[r];
        auto& m = matches12[i];
        m.queryIdx = i;
        ThreadBuf& buf = bufs[tid];

        const uint64_t b1 = binarys1[i];
        const uint64_t* bin2 = binarys2.data();
        uint32_t* costList = buf.costList.data();
        int* costHist = buf.costHist.data();

        std::memset(costHist, 0, 65 * sizeof(int));
        for (int j = 0; j < desc2.rows; ++j) {
#ifdef _WIN32
            const auto c = __popcnt64(b1 ^ bin2[0]);
#else
            const auto c = __builtin_popcountll(b1 ^ bin2[0]);
#endif
            costHist[c]++;
            costList[j] = c;
            bin2 += 1;
        }

        // Keep the cheapest ~kCandidates as the float-computation set,
        // mirroring the SIFT MixMatching window.
        const int kCandidates = 20;
        uint32_t costThreshold = 0;
        uint32_t minCost = 65;
        int histIntegral = 0;
        for (int j = 0; j <= 64; ++j) {
            if (minCost == 65 && costHist[j] > 0) {
                minCost = (uint32_t)j;
            }
            histIntegral += costHist[j];
            if (histIntegral >= kCandidates) {
                costThreshold = (uint32_t)j;
                break;
            }
        }
        costThreshold = std::min(std::max(costThreshold, minCost + 2), minCost + 10);

        const float* ptr1 = desc1.ptr<float>(i);
        float mins = std::numeric_limits<float>::max();
        int bestj = 0;
        for (int j = 0; j < desc2.rows; ++j) {
            if (costList[j] <= costThreshold) {
                const float* ptr2 = desc2.ptr<float>(j);
                const float s = L2Sqr64(ptr1, ptr2);
                if (s < mins) {
                    mins = s;
                    bestj = j;
                }
            }
        }
        m.trainIdx = bestj;
        m.distance = std::sqrt(mins);
    });
}

/**
 * @brief Hybrid binary/float cross matching with mutual best check (mirrors
 *        the SIFT CrossMatching). For full-resolution matching.
 */
void CrossMatchingXFeat(const cv::Mat& desc1, const cv::Mat& desc2,
                        const std::vector<uint64_t>& binarys1,
                        const std::vector<uint64_t>& binarys2,
                        std::vector<cv::DMatch>& matches12,
                        double minCossim) {
    std::vector<bool> flags1(desc1.rows, true);
    std::vector<cv::DMatch> matches1, matches2;
    MixMatchingXFeat(desc1, desc2, binarys1, binarys2, flags1, matches1);

    const double minDist = std::sqrt(std::max(0.0, 2.0 * (1.0 - minCossim)));
    std::vector<bool> flags2(desc2.rows, false);
    for (const auto& m1 : matches1) {
        if (m1.distance < minDist) {
            flags2[m1.trainIdx] = true;
        }
    }

    MixMatchingXFeat(desc2, desc1, binarys2, binarys1, flags2, matches2);

    matches12.clear();
    for (const auto& m1 : matches1) {
        if (flags2[m1.trainIdx] && m1.queryIdx == matches2[m1.trainIdx].trainIdx) {
            cv::DMatch mm = m1;
            // Euclidean distance -> cosine similarity (matching original output).
            mm.distance = 1.0 - 0.5 * m1.distance * m1.distance;
            matches12.push_back(mm);
        }
    }
}

/**
 * @brief Performs mutual best-match between two sets of L2-normalized
 *        descriptors using cosine similarity, mirroring the official XFeat
 *        match() function. Used for low-resolution matching (no binary
 *        descriptors available).
 */
void CrossMatchingXFeat(const cv::Mat& desc1, const cv::Mat& desc2,
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

XFeatExtractor::XFeatExtractor(const std::string& configFile) {
    std::string weightPath = "xfeat.bin";
    std::string onnxPath;
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    if (fs.isOpened()) {
        topK_ = fs["feature"]["xfeat"]["xfeat_top_k"].empty() ? 2000 : (int)fs["feature"]["xfeat"]["xfeat_top_k"];
        matchThreshold_ = fs["feature"]["xfeat"]["xfeat_match_threshold"].empty() ? 0.82 : (double)fs["feature"]["xfeat"]["xfeat_match_threshold"];
        weightPath = fs["feature"]["xfeat"]["xfeat_weight"].empty() ? std::string("xfeat.bin") : (std::string)fs["feature"]["xfeat"]["xfeat_weight"];
        onnxPath = fs["feature"]["xfeat"]["xfeat_onnx_weight"].empty() ? std::string() : (std::string)fs["feature"]["xfeat"]["xfeat_onnx_weight"];
        fs.release();
    }

    xfeat_ = std::make_shared<XFeat>();
    if (!onnxPath.empty()) {
        if (xfeat_->LoadONNX(onnxPath)) {
            return;
        }
        std::cerr << "XFeatExtractor: failed to load XFeat ONNX model: " << onnxPath
                  << ", falling back to hand-written engine" << std::endl;
    }
    if (!xfeat_->Load(weightPath)) {
        std::cerr << "XFeatExtractor: failed to load XFeat weights: " << weightPath << std::endl;
        xfeat_.reset();
    }
}

bool XFeatExtractor::Extract(const cv::Mat& image,
                             std::vector<cv::KeyPoint>& keypoints,
                             cv::Mat& descriptors,
                             std::vector<uint64_t>& binaryDescriptors,
                             std::vector<float>& scores) {
    std::cout << "use xfeat\n";
    binaryDescriptors.clear();
    if (!xfeat_) {
        return false;
    }
    if (!xfeat_->detectAndCompute(image, topK_, keypoints, descriptors, scores)) {
        return false;
    }
    ToBinaryDescXFeat(descriptors, binaryDescriptors);
    return true;
}

bool XFeatExtractor::ExtractLowRes(const cv::Mat& image,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   cv::Mat& descriptors) {
    if (!xfeat_) {
        return false;
    }
    std::vector<float> scores;
    return xfeat_->detectAndCompute(image, 300, keypoints, descriptors, scores);
}

void XFeatExtractor::Match(const cv::Mat& desc1, const std::vector<uint64_t>& binary1,
                           const cv::Mat& desc2, const std::vector<uint64_t>& binary2,
                           std::vector<cv::DMatch>& matches) const {
    if (binary1.empty() || binary2.empty() ||
        binary1.size() != (size_t)desc1.rows || binary2.size() != (size_t)desc2.rows) {
        CrossMatchingXFeat(desc1, desc2, matches, matchThreshold_);
        return;
    }
    CrossMatchingXFeat(desc1, desc2, binary1, binary2, matches, matchThreshold_);
}

void XFeatExtractor::MatchLowRes(const cv::Mat& desc1, const cv::Mat& desc2,
                                 std::vector<cv::DMatch>& matches) const {
    CrossMatchingXFeat(desc1, desc2, matches, matchThreshold_);
}

}  // namespace sfm
