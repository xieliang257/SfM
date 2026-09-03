#include "FeatureExtractor/SiftExtractor.h"
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
 * @brief Extracts keypoints from an image using SIFT and distributes them
 *        uniformly across the image (unchanged from the original pipeline).
 */
void ExtractUniformKeypoints(const cv::Mat& image, int blocks, int kptCnt, double quality,
                             std::vector<cv::KeyPoint>& kpts) {
    auto sift = cv::SIFT::create(kptCnt * 10, 3, quality, 10);
    sift->detect(image, kpts);

    std::sort(kpts.begin(), kpts.end(), [](cv::KeyPoint& k1, cv::KeyPoint& k2) {return k1.response > k2.response; });

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    int idd = 0;
    for (int i = 0; i < kpts.size(); ++i) {
        int x = kpts[i].pt.x, y = kpts[i].pt.y;
        if (x > 0 && x < mask.cols - 1 && y>0 && y < mask.rows - 1 && mask.at<uchar>(y, x) == 0) {
            kpts[idd] = kpts[i];
            ++idd;
            mask.at<uchar>(y, x) = 255;
            for (int v = y - 1; v <= y + 1; ++v) {
                for (int u = x - 1; u <= x + 1; ++u) {
                    mask.at<uchar>(v, u) = 255;
                }
            }
        }
    }
    kpts.resize(idd);

    int w = sqrt(image.rows * image.cols / blocks);

    std::vector<std::vector<std::vector<cv::KeyPoint>>> blockKeypts;
    blockKeypts.resize(image.rows / w + 1);
    for (auto& b : blockKeypts) {
        b.resize(image.cols / w + 1);
    }

    std::vector<cv::KeyPoint> otherPts;

    for (auto& k : kpts) {
        int i = k.pt.y / w;
        int j = k.pt.x / w;

        if (blockKeypts[i][j].size() <= kptCnt / blocks) {
            blockKeypts[i][j].push_back(k);
        }
        else {
            otherPts.push_back(k);
        }
    }

    kpts.clear();
    for (int i = 0; i < blockKeypts.size(); ++i) {
        for (int j = 0; j < blockKeypts[i].size(); ++j) {
            kpts.insert(kpts.end(), blockKeypts[i][j].begin(), blockKeypts[i][j].end());
        }
    }

    if (kpts.size() > kptCnt) {
        kpts.resize(kptCnt);
        return;
    }

    if (kpts.size() < kptCnt) {
        for (auto& other : otherPts) {
            kpts.push_back(other);
            if (kpts.size() >= kptCnt) {
                break;
            }
        }
    }
}

/**
 * @brief Converts floating-point descriptors to a binary format for efficient
 *        matching (unchanged from the original pipeline).
 */
void ToBinaryDesc(const cv::Mat& descMat_, std::vector<uint64_t>& binaryDesc) {
    binaryDesc.resize(descMat_.rows * 2);
    for (int i = 0; i < descMat_.rows; ++i) {
        double avg = cv::mean(descMat_.row(i))[0];
        uint64_t b1 = 0, b2 = 0;
        float* ptr1 = (float*)descMat_.ptr(i);
        float* ptr2 = ptr1 + 64;
        for (int j = 0; j < 64; ++j) {
            b1 <<= 1;
            b2 <<= 1;
            if (ptr1[j] > avg) {
                b1++;
            }
            if (ptr2[j] > avg) {
                b2++;
            }
        }
        binaryDesc[i * 2] = b1;
        binaryDesc[i * 2 + 1] = b2;
    }
}

/**
 * @brief Pure float brute-force cross matching with mutual best check
 *        (unchanged from the original pipeline).
 */
void CrossMatching(const cv::Ptr<cv::DescriptorMatcher>& matcher, const cv::Mat& desc1, const cv::Mat& desc2,
    std::vector<cv::DMatch>& match_ij_out, double minDistance) {
    std::vector<cv::DMatch> match_ij, match_ji;
    matcher->match(desc1, desc2, match_ij);
    matcher->match(desc2, desc1, match_ji);

    int inlierId = 0;
    for (int k = 0; k < match_ij.size(); ++k) {
        int qId = match_ij[k].queryIdx;
        int tId = match_ij[k].trainIdx;
        if (tId < match_ji.size() && qId == match_ji[tId].trainIdx && match_ij[k].distance < minDistance) {
            match_ij[inlierId] = match_ij[k];
            ++inlierId;
        }
    }
    match_ij.resize(inlierId);
    match_ij_out = match_ij;
}

/**
 * @brief Mixed matching combining binary and floating-point descriptors
 *        (unchanged from the original pipeline).
 */
void MixMatching(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<uint64_t>& binarys1,
    const std::vector<uint64_t>& binarys2, const std::vector<bool>& flags, std::vector<cv::DMatch>& matches12) {
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
        explicit ThreadBuf(int row2) : costList(row2), costHist(129, 0) {}
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

        const uint64_t b1 = binarys1[i * 2];
        const uint64_t b2 = binarys1[i * 2 + 1];
        const uint64_t* bin2 = binarys2.data();
        uint32_t* costList = buf.costList.data();
        int* costHist = buf.costHist.data();

        std::memset(costHist, 0, 129 * sizeof(int));
        for (int j = 0; j < desc2.rows; ++j) {
#ifdef _WIN32
            const auto c = __popcnt64(b1 ^ bin2[0]) + __popcnt64(b2 ^ bin2[1]);
#else
            const auto c = __builtin_popcountll(b1 ^ bin2[0]) + __builtin_popcountll(b2 ^ bin2[1]);
#endif
            costHist[c]++;
            costList[j] = c;
            bin2 += 2;
        }

        uint32_t costThreshold = 0;
        uint32_t minCost = 129;
        int histIntegral = 0;
        for (int j = 0; j <= 128; ++j) {
            if (minCost == 129 && costHist[j] > 0) {
                minCost = (uint32_t)j;
            }
            histIntegral += costHist[j];
            if (histIntegral >= 20) {
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
                const float s = L2Sqr128(ptr1, ptr2);
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
 * @brief Hybrid binary/float cross matching (unchanged from the original
 *        pipeline). Used for full-resolution matching.
 */
void CrossMatching(const cv::Mat& desc1,
                   const cv::Mat& desc2,
                   const std::vector<uint64_t>& binarys1,
                   const std::vector<uint64_t>& binarys2,
                   std::vector<cv::DMatch>& matches12,
                   double threshold) {

    std::vector<bool> flags1(desc1.rows, true);
    std::vector<cv::DMatch> matches1, matches2;
    MixMatching(desc1, desc2, binarys1, binarys2, flags1, matches1);

    std::vector<bool> flags2(desc2.rows, false);
    for (auto& m1 : matches1) {
        if (m1.distance < threshold) {
            flags2[m1.trainIdx] = true;
        }
    }

    MixMatching(desc2, desc1, binarys2, binarys1, flags2, matches2);

    matches12.clear();
    for (auto& m1 : matches1) {
        if (flags2[m1.trainIdx] && m1.queryIdx == matches2[m1.trainIdx].trainIdx) {
            matches12.push_back(m1);
        }
    }
}

}  // namespace

SiftExtractor::SiftExtractor(const std::string& configFile) {
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    if (fs.isOpened()) {
        featureCount_ = fs["feature"]["sift"]["feature_count"].empty() ? 12000 : (int)fs["feature"]["sift"]["feature_count"];
        fs.release();
    }
    sift_ = cv::SIFT::create(featureCount_);
}

bool SiftExtractor::Extract(const cv::Mat& image,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& descriptors,
                            std::vector<uint64_t>& binaryDescriptors,
                            std::vector<float>& scores) {
    std::cout << "use sift\n";
    ExtractUniformKeypoints(image, 100, featureCount_, 0.005, keypoints);
    sift_->compute(image, keypoints, descriptors);

    for (int i = 0; i < descriptors.rows; ++i) {
        descriptors.row(i) /= cv::norm(descriptors.row(i));
    }

    ToBinaryDesc(descriptors, binaryDescriptors);
    return !keypoints.empty();
}

bool SiftExtractor::ExtractLowRes(const cv::Mat& image,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors) {
    ExtractUniformKeypoints(image, 100, 300, 0.005, keypoints);
    sift_->compute(image, keypoints, descriptors);

    for (int i = 0; i < descriptors.rows; ++i) {
        descriptors.row(i) /= cv::norm(descriptors.row(i));
    }
    return !keypoints.empty();
}

void SiftExtractor::Match(const cv::Mat& desc1, const std::vector<uint64_t>& binary1,
                          const cv::Mat& desc2, const std::vector<uint64_t>& binary2,
                          std::vector<cv::DMatch>& matches) const {
    CrossMatching(desc1, desc2, binary1, binary2, matches, 0.5);
}

void SiftExtractor::MatchLowRes(const cv::Mat& desc1, const cv::Mat& desc2,
                                std::vector<cv::DMatch>& matches) const {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    CrossMatching(matcher, desc1, desc2, matches, 0.6);
}

}  // namespace sfm
