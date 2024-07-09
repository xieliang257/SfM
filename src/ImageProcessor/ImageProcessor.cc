#include <iostream>
#include <io.h>
#include <fstream>
#include <direct.h>
#include <thread>
#include "ImageProcessor/ImageProcessor.h"

namespace sfm {
void CollectImagePath(const std::string& dirPath, std::string format, std::vector<std::string>& files);

void RemoveByMatchGraph(AllMatchesType& matches);

void CrossMatching(const cv::Ptr<cv::DescriptorMatcher>& matcher, const cv::Mat& desc1, const cv::Mat& desc2, 
    std::vector<cv::DMatch>& match_ij_out, double minDistance);

void CrossMatching(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<uint64_t>& binarys1,
    const std::vector<uint64_t>& binarys2, std::vector<cv::DMatch>& matches12, double threshold);

void MixMatching(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<uint64_t>& binarys1,
    const std::vector<uint64_t>& binarys2, const std::vector<bool>& flags, std::vector<cv::DMatch>& matches12);

void SimilarityRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2, double trainThreshold, double testThreshold);

void FundamentalRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2);

ImageProcessor::ImageProcessor(const std::string& configFile, const std::string& workDir) {
    workDir_ = workDir;
    configFile_ = configFile;
    pFrames_ = std::make_shared<std::vector<Frame>>();
    pMatches_ = std::make_shared<AllMatchesType>();
}

const std::shared_ptr<std::vector<Frame>>& ImageProcessor::FramePtr() {
    return pFrames_;
}

const std::shared_ptr<AllMatchesType>& ImageProcessor::MatchesPtr() {
    return pMatches_;
}

/**
 * @brief Extracts features from images and matches them across frames.
 *
 * This function handles the complete image processing sequence for a given directory of images,
 * which includes loading previously processed frames and matches, collecting image paths, sorting them,
 * and extracting features from each image. After the feature extraction, it performs matching of features
 * across all frames. Finally, it saves the extracted frame data and match results for later use. The function
 * also creates a directory for storing undistorted images. Execution time for feature extraction is
 * recorded and displayed, providing insights into the performance of the extraction process.
 *
 * @param imgDir Directory containing the images to be processed.
 */
void ImageProcessor::ExtractAndMatchAll(const std::string& imgDir) {
    // Read previously processed frame data and matching data from a file.
    std::string pathToSave = workDir_ + "/Match";
    ReadFrames(pathToSave + "/Frames.txt", pFrames_);
    ReadMatches(pathToSave + "/Matches.txt", pMatches_);

    // Check if the frames and matches are already loaded and valid; if so, exit early.
    if (pFrames_ && pMatches_ && pFrames_->size() > 0 && pMatches_->size() > 0) {
        return;
    }

    // Collect paths of images in the specified directory with png and jpg extensions.
    std::vector<std::string> imgPaths;
    CollectImagePath(imgDir + "/", "*.png", imgPaths);
    CollectImagePath(imgDir + "/", "*.jpg", imgPaths);

    // Define a lambda function to compare two image paths based on their size and lexicographical order.
    auto PathCompare = [](const std::string& s1, const std::string& s2) {
        if (s1.size() == s2.size()) {
            return s1 < s2;
        }
        return s1.size() < s2.size();
    };

    // Sort the image paths using the defined comparison function.
    std::sort(imgPaths.begin(), imgPaths.end(), PathCompare);

    // Extract features from each image in the sorted list of image paths.
    auto t0 = cv::getTickCount();
    int idx = 0;
    for (const auto& path : imgPaths) {
        ++idx;
        std::cout << "\rExtracting (" << idx << "/" << imgPaths.size() << "): " << path << "  ";
        Frame frame(configFile_, workDir_);
        bool flag = frame.LoadAndExtract(path);
        pFrames_->push_back(frame);
        auto t1 = cv::getTickCount();
        double fs = 1. / cv::getTickFrequency();
        std::cout << "Keypoints: " << frame.keypointList_.size() << "  Cost: " << (t1 - t0) * fs << " s    ";
    }
    std::cout << "\n";

    // Match all extracted frames to find correspondences between them.
    MatchAll();

    // Save the frames and matches back to files for future use.
    mkdir(pathToSave.c_str());
    SaveFrames(workDir_ + "/Match/Frames.txt", pFrames_);
    SaveMatches(workDir_ + "/Match/Matches.txt", pMatches_);
}

void ImageProcessor::ReadMatches(const std::string& path, std::shared_ptr<AllMatchesType>& pMatches) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return;
    }
    int size;
    in >> size;
    pMatches->resize(size);
    for (auto& m : *pMatches) {
        m.resize(size);
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int n = 0;
            in >> n;
            if (n > 0) {
                std::vector<cv::DMatch> dms;
                for (int k = 0; k < n; ++k) {
                    cv::DMatch m;
                    in >> m.queryIdx >> m.trainIdx;
                    dms.push_back(m);
                }
                (*pMatches)[i][j] = dms;
            }
        }
    }
    in.close();
}

void ImageProcessor::ReadFrames(const std::string& path, std::shared_ptr<std::vector<Frame>>& pFrames) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return;
    }
    int frameCnt = 0;
    in >> frameCnt;
    pFrames->resize(frameCnt);
    for (int i = 0; i < frameCnt; ++i) {
        Frame frame(configFile_, workDir_);
        in >> frame.imagePath_;
        int kpts;
        in >> kpts;
        for (int j = 0; j < kpts; ++j) {
            cv::KeyPoint kpt;
            float x, y;
            in >> x >> y;
            kpt.pt.x = x;
            kpt.pt.y = y;
            frame.keypointList_.push_back(kpt);
        }
        for (int j = 0; j < kpts; ++j) {
            int r, g, b;
            in >> r >> g >> b;
            frame.colorList_.push_back(cv::Vec3b(r, g, b));
        }
        (*pFrames)[i] = frame;
    }
    in.close();
}

void ImageProcessor::SaveMatches(const std::string& path, const std::shared_ptr<AllMatchesType>& pMatches) {
    std::ofstream out(path);
    out << pMatches->size() << "\n";
    for (int i = 0; i < pMatches->size(); ++i) {
        for (int j = 0; j < (*pMatches)[i].size(); ++j) {
            out << (*pMatches)[i][j].size() << "\n";
            for (const auto& m : (*pMatches)[i][j]) {
                out << m.queryIdx << " " << m.trainIdx << " ";
            }
            out << "\n";
        }
    }
    out.close();
}

void ImageProcessor::SaveFrames(const std::string& path, const std::shared_ptr<std::vector<Frame>>& pFrames) {
    const auto& frames = *pFrames;
    std::ofstream out(path);
    out << frames.size() << "\n";
    for (const auto& frame : frames) {
        out << frame.imagePath_ << "\n";
        out << frame.keypointList_.size() << "\n";
        for (const auto& kpt : frame.keypointList_) {
            out << float(kpt.pt.x) << " " << float(kpt.pt.y) << "\n";
        }
        for (const auto& color : frame.colorList_) {
            out << int(color[0]) << " " << int(color[1]) << " " << int(color[2]) << "\n";
        }
    }
    out.close();
}

void CollectImagePath(const std::string& dirPath, std::string format, std::vector<std::string>& files) {
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;
    std::string p;
    if ((hFile = _findfirst(p.assign(dirPath).append("\\*" + format).c_str(), &fileInfo)) != -1) {
        do {
            files.push_back(p.assign(dirPath).append("\\").append(fileInfo.name));
        } while (_findnext(hFile, &fileInfo) == 0);
        _findclose(hFile);
    }
}

/**
 * @brief Matches features across all frames to build a comprehensive set of inter-frame correspondences.
 *
 * This function orchestrates the matching of features across all available frames. It begins by employing a brute force
 * matcher using a limited number of keypoints to rapidly generate a preliminary match graph. This match graph determines
 * which frame pairs should be processed for extensive matching. Only frame pairs marked in this match graph undergo further
 * detailed matching using the full set of descriptors. The detailed matching process involves cross-correlation of descriptor
 * lists and employs RANSAC algorithms to filter out outliers based on geometric consistency. Multiple rounds of filtering,
 * including similarity-based and Fundamental-based RANSAC, are applied to ensure reliable matching.
 *
 * Operates on internal member variables `pMatches_` and `pFrames_`, which should be pre-initialized and filled with data
 * respectively. These are assumed to be pointers to a vector of `Frame` objects and a vector of match lists
 * (vector<vector<cv::DMatch>>), typically set up prior to this function call.
 */
void ImageProcessor::MatchAll() {
    AllMatchesType& matches = *pMatches_;
    std::vector<Frame>& frames = *pFrames_;
    matches.resize(frames.size());
    for (int i = 0; i < matches.size(); ++i) {
        matches[i].resize(frames.size());
    }

    // Create a brute force matcher.
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    // Initialize a match graph matrix with zero indicating no match between frames.
    cv::Mat matchGraph = cv::Mat::zeros(frames.size(), frames.size(), CV_32S);

    // Populate the match graph using a subset of keypoints at a lower resolution to perform initial matching. 
    // This process involves using RANSAC to remove outliers and establish a reliable matching set. 
    // The relationship between two images is determined based on the count of successfully matched points.
    BuildMatchGraph(matchGraph);

    auto t0 = cv::getTickCount();
    int matchCnt = 0;

    // Loop over all possible pairs of frames to find matches.
    for (int i = 0; i < frames.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            // Skip if no match is indicated in the graph.
            if (matchGraph.at<int>(i, j) == 0) {
                continue;
            }
            ++matchCnt;
            auto t1 = cv::getTickCount();
            double tcost = double(t1 - t0) / cv::getTickFrequency();
            std::cout << "\rMatching " << i << ", " << j << "  Total Matched: " << matchCnt << " pairs  Cost: " << tcost << " s   ";
            std::vector<cv::DMatch> match_ij, tmpij;
            // Perform cross matching between descriptors of two frames.
            CrossMatching(frames[i].descList_, frames[j].descList_, frames[i].binaryDescs_, frames[j].binaryDescs_, match_ij, 0.5);

            // Check if there are enough matches.
            if (match_ij.size() < 30) {
                continue;
            }

            // Apply RANSAC to filter out unreliable matches based on geometric consistency.
            double highTrainThreshold = sqrt(frames[i].Width() * frames[i].Height()) * 0.1;
            double highTestThreshold = sqrt(frames[i].Width() * frames[i].Height()) * 0.2;
            int beforeCnt = match_ij.size();
            SimilarityRansac(match_ij, frames[i].keypointList_, frames[j].keypointList_, highTrainThreshold, highTestThreshold);
            if (match_ij.size() < 30) {
                continue;
            }

            FundamentalRansac(match_ij, frames[i].keypointList_, frames[j].keypointList_);
            if (match_ij.size() < 30) {
                continue;
            }
            int afterCnt = match_ij.size();
            if (afterCnt < beforeCnt * 0.03) {
                match_ij.clear();
                continue;
            }

            // Store the final set of matches.
            matches[i][j] = match_ij;
        }
    }
    std::cout << "\n";
    RemoveByMatchGraph(matches);
}

/**
 * @brief Constructs a match graph based on initial matches between low-resolution keypoints of all frames.
 *
 * This function generates a match graph to record potential matches between every pair of frames in the dataset.
 * It employs a brute force matcher to perform initial matches between a set of low-resolution descriptors from each frame.
 * These initial matches are further refined using a RANSAC algorithm to exclude outliers and ensure the matches are geometrically
 * consistent. 
 * The graph is populated based on the count of matches that pass the RANSAC filtering.
 * A threshold is set to determine if enough matches exist between two frames to consider them connected in the graph. 
 * The match count is used as the weight in the graph, indicating the strength of the connection between frame pairs.
 *
 * The match graph is a symmetric matrix where the element at (i, j) indicates the number of reliable matches between frame i and
 * frame j. 
 * If the number of matches is below a set threshold (in this case, 10), the frames are considered not to have a significant
 * relationship, and the corresponding graph element is set to zero.
 *
 * @param matchGraph Reference to a cv::Mat object where the match graph will be stored. The matrix is initialized within this function.
 */
void ImageProcessor::BuildMatchGraph(cv::Mat& matchGraph) {
    std::vector<Frame>& frames = *pFrames_;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    auto t0 = cv::getTickCount();

    // Initialize the match graph as a zero matrix of size equal to the number of frames.
    matchGraph = cv::Mat::zeros(frames.size(), frames.size(), CV_32S);
    for (int i = 0; i < frames.size(); ++i) {
        auto t1 = cv::getTickCount();
        double tcost = double(t1 - t0) / cv::getTickFrequency();
        std::cout << "\rBuilding match graph: " << i << "/" << frames.size() << "  Cost: " << tcost <<" s   ";
        for (int j = 0; j < i; ++j) {
            std::vector<cv::DMatch> match_ij, low_match_ij;
            // Perform cross matching between low-resolution descriptors with a specific ratio.
            CrossMatching(matcher, frames[i].lDescList_, frames[j].lDescList_, low_match_ij, 0.6);

            // Apply RANSAC to filter out outlier matches using specified thresholds.
            double lowTrainThreshold = frames[i].LowResWidth() * 0.1;
            double lowTestThreshold = frames[i].LowResWidth() * 0.1;
            SimilarityRansac(low_match_ij, frames[i].lKeyPts_, frames[j].lKeyPts_, lowTrainThreshold, lowTestThreshold);
            if (low_match_ij.size() < 10) {
                continue;
            }

            // Symmetrically update the match graph with the count of matches.
            matchGraph.at<uint>(i, j) = matchGraph.at<uint>(j, i) = low_match_ij.size();
        }
    }
    std::cout << "\n";
}

/**
 * @brief Refines the match data based on the match graph, removing matches that do not meet a certain threshold of significance.
 *
 * This function constructs a match graph that maps the number of matches between every pair of frames and then refines this
 * graph to only retain significant matches. Each entry in the graph starts as the number of matches between two frames. The function
 * then calculates a significance threshold for each row of the match graph, which is set as the minimum of one-fifth of the maximum
 * match count or the fifth highest match count in that row. Matches that do not exceed this threshold are considered insignificant
 * and are removed from the graph. The final stage involves clearing the actual match data for pairs that no longer meet the required
 * threshold of matches as determined by the refined match graph.
 *
 * @param matches Reference to a container that stores all matches between frames. It is modified in place, with non-significant
 *                matches being cleared.
 */
void RemoveByMatchGraph(AllMatchesType& matches) {
    cv::Mat matchGraph = cv::Mat::zeros(matches.size(), matches.size(), CV_32S);
    // Populate the graph with the count of matches for each frame pair.
    for (int i = 0; i < matches.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            int n = matches[i][j].size();
            if (n > 0) {
                matchGraph.at<_int32>(i, j) = matchGraph.at<_int32>(j, i) = n;
            }
        }
    }

    // Determine the significance threshold for each row in the match graph.
    for (int i = 0; i < matchGraph.rows; ++i) {
        _int32 maxCnt = 0;
        std::vector<_int32> cntList;
        for (int j = 0; j < matchGraph.cols; ++j) {
            maxCnt = std::max(maxCnt, matchGraph.at<_int32>(i, j));
            cntList.push_back(matchGraph.at<_int32>(i, j));
        }
        std::sort(cntList.begin(), cntList.end());
        // Set threshold to one-fifth of the maximum count.
        int cntThreshold = maxCnt / 5;

        // Adjust threshold based on the sorted list of counts.
        if (cntList.size() <= 5) {
            //cntThreshold = 0;
        }
        else {
            cntThreshold = std::min(cntThreshold, cntList[cntList.size() - 5]);
        }

        // Apply the threshold, discarding matches that do not meet the criteria.
        for (int j = 0; j < matchGraph.cols; ++j) {
            if (matchGraph.at<_int32>(i, j) < cntThreshold) {
                matchGraph.at<_int32>(i, j) = 0;
            }
        }
    }

    // Ensure the match graph is symmetric.
    for (int i = 0; i < matchGraph.rows; ++i) {
        for (int j = 0; j < matchGraph.cols; ++j) {
            matchGraph.at<_int32>(i, j) = matchGraph.at<_int32>(j, i) =
                std::max(matchGraph.at<_int32>(i, j), matchGraph.at<_int32>(j, i));
        }
    }

    // Clear the match lists based on the updated match graph values.
    for (int i = 0; i < matches.size(); ++i) {
        for (int j = 0; j < matches[i].size(); ++j) {
            if (matches[i][j].size() > matchGraph.at<_int32>(i, j)) {
                matches[i][j].clear();
            }
        }
    }
}

/**
 * @brief Performs cross-matching between two sets of descriptors to find mutual best matches.
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
 * @brief Performs robust cross-matching between two sets of descriptors using both traditional and binary descriptors.
 *
 * This function cross-matches descriptors from two datasets (desc1 and desc2) ensuring mutual consistency and filtering
 * based on a distance threshold. It employs an initial matching from desc1 to desc2, then verifies these matches by
 * ensuring they are the best matches in the opposite direction from desc2 to desc1. The binary descriptors are used
 * to quickly eliminate non-matching descriptors before a more detailed check is done. Matches that pass the threshold
 * check and have reciprocal best matches are considered valid and are stored in the output.
 *
 * @param desc1 Descriptors from the first dataset.
 * @param desc2 Descriptors from the second dataset.
 * @param binarys1 Binary descriptors corresponding to desc1.
 * @param binarys2 Binary descriptors corresponding to desc2.
 * @param matches12 Output vector to store the refined matches.
 * @param threshold The maximum allowed distance for matches to be considered valid.
 */
void CrossMatching(const cv::Mat& desc1,
                   const cv::Mat& desc2,
                   const std::vector<uint64_t>& binarys1,
                   const std::vector<uint64_t>& binarys2,
                   std::vector<cv::DMatch>& matches12,
                   double threshold) {

    // Initial flags to track whether matches from desc1 meet the threshold criteria.
    std::vector<bool> flags1(desc1.rows, true);
    std::vector<cv::DMatch> matches1, matches2;
    // Perform initial matching using descriptors and binary descriptors.
    MixMatching(desc1, desc2, binarys1, binarys2, flags1, matches1);

    // Flags to mark descriptors in desc2 that meet the match criteria.
    std::vector<bool> flags2(desc2.rows, false);
    for (auto& m1 : matches1) {
        if (m1.distance < threshold) {
            flags2[m1.trainIdx] = true;
        }
    }

    // Verify the initial matches by performing matching in the reverse direction.
    MixMatching(desc2, desc1, binarys2, binarys1, flags2, matches2);

    // Cross check
    matches12.clear();
    for (auto& m1 : matches1) {
        if (flags2[m1.trainIdx] && m1.queryIdx == matches2[m1.trainIdx].trainIdx) {
            matches12.push_back(m1);
        }
    }
}

/**
 * @brief Performs feature matching between two descriptor sets using binary and float-based descriptors.
 *
 * This function executes a mixed matching process that combines binary and floating-point descriptor comparisons
 * to refine match quality. Initially, binary descriptors are used to quickly assess and filter potential matches based
 * on a Hamming distance threshold. The most promising matches are then evaluated more rigorously using the Euclidean
 * distance between traditional floating-point descriptors.
 *
 * The function is optimized for performance with multi-threading, dividing the matching process across several threads
 * to leverage modern CPU architectures.
 *
 * @param desc1 Floating-point descriptors for the first image.
 * @param desc2 Floating-point descriptors for the second image.
 * @param binarys1 Binary descriptors corresponding to desc1.
 * @param binarys2 Binary descriptors corresponding to desc2.
 * @param flags A vector of booleans indicating which descriptors in desc1 are eligible for matching.
 * @param matches12 Output vector to store the results of the matching process.
 */
void MixMatching(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<uint64_t>& binarys1,
    const std::vector<uint64_t>& binarys2, const std::vector<bool>& flags, std::vector<cv::DMatch>& matches12) {
    matches12.resize(desc1.rows);
    // Lambda function to process a segment of descriptors.
    auto Func = [&](int start, int end) {
        int row2 = desc2.rows;
        // Cost list based on binary descriptor comparison.
        std::vector<uint32_t> costList(row2);
        for (int i = start; i < end; ++i) {
            // Skip descriptors not flagged for matching.
            if (!flags[i]) {
                continue;
            }
            auto& m = matches12[i];
            m.queryIdx = i;

            // Retrieve binary descriptors for comparison.
            auto b1 = binarys1[i * 2];
            auto b2 = binarys1[i * 2 + 1];
            auto bin2 = binarys2.data();

            // Histogram of Hamming distances.
            std::vector<int> costHist(129, 0);
            for (int j = 0; j < row2; ++j) {
                // Compute Hamming distance.
                auto c = __popcnt64(b1 ^ bin2[0]) + __popcnt64(b2 ^ bin2[1]);
                costHist[c]++;
                costList[j] = c;
                bin2 += 2;
            }

            // Determine the threshold for switching to float comparison.
            uint32_t costThreshold = 0;
            uint32_t minCost = 129;
            int histIntegral = 0;
            for (int j = 0; j <= 128; ++j) {
                if (minCost == 129 && costHist[j] > 0) {
                    minCost = j;
                }
                histIntegral += costHist[j];
                // Threshold based on cumulative histogram.
                if (histIntegral >= 20) {
                    costThreshold = j;
                    break;
                }
            }

            costThreshold = std::min(std::max(costThreshold, minCost + 2), minCost + 10);
            auto ptr1 = desc1.ptr<float>(i);
            float mins = std::numeric_limits<float>::max();
            int bestj = 0;
            for (int j = 0; j < row2; ++j) {
                // Check against the refined threshold.
                if (costList[j] <= costThreshold) {
                    auto ptr2 = desc2.ptr<float>(j);

                    // Squared Euclidean distance.
                    float s = 0;
                    for (int k = 0; k < 128; ++k) {
                        auto d = ptr1[k] - ptr2[k];
                        s += d * d;
                    }
                    // Find the descriptor with the minimum distance.
                    if (s < mins) {
                        mins = s;
                        bestj = j;
                    }

                }
            }

            // Record the best match found and compute and store the Euclidean distance.
            m.trainIdx = bestj;
            m.distance = sqrt(mins);
        }
    };

    // Apply threads to process the matching in parallel.
    std::vector<std::thread> threads;
    int numThreads = std::thread::hardware_concurrency();
    int step = (desc1.rows + numThreads - 1) / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * step;
        int end = std::min((i + 1) * step, desc1.rows);
        threads.emplace_back(Func, start, end);
    }

    // Wait for all threads to complete.
    for (auto& t : threads) {
        t.join();
    }
}

/**
 * @brief Performs a RANSAC-based estimation of similarity transformation between matched keypoints from two images.
 *
 * @param matches Input and output vector of matches between keypoints; refined by removing outliers.
 * @param kpts1 Keypoints from the first image corresponding to query indices in 'matches'.
 * @param kpts2 Keypoints from the second image corresponding to train indices in 'matches'.
 * @param trainThreshold Distance threshold for counting inliers during model estimation.
 * @param testThreshold Distance threshold for filtering matches when applying the best model.
 */
void SimilarityRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2, double trainThreshold, double testThreshold) {
    int maxIterCnt = 1000;
    cv::RNG rng;
    int maxInliers = 0;
    double bestScale = 1, bestCosVal = 1, bestSinVal = 0, bestTx = 0, bestTy = 0;
    for (int iter = 0; iter < maxIterCnt; ++iter) {
        int id1 = rng.next() % matches.size();
        int id2 = rng.next() % matches.size();
        if (id1 == id2) {
            continue;
        }
        cv::Point2f x1 = kpts1[matches[id1].queryIdx].pt;
        cv::Point2f x2 = kpts1[matches[id2].queryIdx].pt;
        cv::Point2f y1 = kpts2[matches[id1].trainIdx].pt;
        cv::Point2f y2 = kpts2[matches[id2].trainIdx].pt;
        double epsilon = 1e-8;
        cv::Point2f a = x2 - x1, b = y2 - y1;
        double s = (cv::norm(b) + epsilon) / (cv::norm(a) + epsilon);
        if (s < 0.5 || s > 2) {
            continue;
        }
        double cosVal = a.dot(b) / (cv::norm(a) * cv::norm(b) + epsilon);
        double sinVal = sqrt(1 - cosVal * cosVal);
        if (a.x * b.y - a.y * b.x > 0) {
            sinVal = -sinVal;
        }
        double tx = y1.x - s * (cosVal * x1.x + sinVal * x1.y);
        double ty = y1.y - s * (-sinVal * x1.x + cosVal * x1.y);
        int inliers = 0;
        for (const auto& m : matches) {
            cv::Point2f x1 = kpts1[m.queryIdx].pt;
            cv::Point2f y1 = kpts2[m.trainIdx].pt;
            double err_x = s * (cosVal * x1.x + sinVal * x1.y) + tx - y1.x;
            double err_y = s * (-sinVal * x1.x + cosVal * x1.y) + ty - y1.y;
            if (cv::norm(cv::Point2d(err_x, err_y)) < trainThreshold) {
                ++inliers;
            }
        }
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestScale = s;
            bestCosVal = cosVal;
            bestSinVal = sinVal;
            bestTx = tx;
            bestTy = ty;
        }
        double p = double(maxInliers) / matches.size();
        p = std::max(std::min(p, 0.999), 0.001);
        double enoughIters = log10(0.001) / log10((1 - p) * (1 - p));
        if (iter > enoughIters + 100) {
            break;
        }
    }

    int id = 0;
    for (int i = 0; i < matches.size(); ++i) {
        cv::Point2f x1 = kpts1[matches[i].queryIdx].pt;
        cv::Point2f y1 = kpts2[matches[i].trainIdx].pt;
        double err_x = bestScale * (bestCosVal * x1.x + bestSinVal * x1.y) + bestTx - y1.x;
        double err_y = bestScale * (-bestSinVal * x1.x + bestCosVal * x1.y) + bestTy - y1.y;
        if (cv::norm(cv::Point2d(err_x, err_y)) < testThreshold) {
            matches[id] = matches[i];
            ++id;
        }
    }
    matches.resize(id);
    if (id < matches.size() * 0.1) {
        matches.clear();
    }
}

/**
 * @brief Estimates the fundamental matrix and filters out outlier matches using the RANSAC method.
 *
 * @param matches Input and output vector of matches between keypoints; outliers are removed.
 * @param kpts1 Keypoints from the first image corresponding to query indices in 'matches'.
 * @param kpts2 Keypoints from the second image corresponding to train indices in 'matches'.
 */
void FundamentalRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2) {
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }
    double reprojErrThreshold = 10.;
    std::vector<uchar> status;
    cv::findFundamentalMat(pts1, pts2, status, cv::RANSAC, reprojErrThreshold);
    auto fInliers = std::count_if(status.begin(), status.end(), [](uchar s) { return s != 0; });

    int threshold = std::min(pts1.size() * 0.2, 50.);
    if (fInliers < threshold) {
        matches.clear();
        return;
    }

    int id = 0;
    for (int i = 0; i < matches.size(); ++i) {
        if (status[i]) {
            matches[id] = matches[i];
            ++id;
        }
    }
    if (id < 30) {
        id = 0;
    }
    matches.resize(id);
}

}