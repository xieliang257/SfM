#include <iostream>
#include <fstream>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <fnmatch.h>
#include <cstdint>
#endif

#include "ImageProcessor/ImageProcessor.h"
#include "FeatureExtractor/SiftExtractor.h"
#include "FeatureExtractor/XFeatExtractor.h"
#include "FeatureExtractor/SuperPointExtractor.h"

namespace sfm {
void CollectImagePath(const std::string& dirPath, std::string format, std::vector<std::string>& files);

void RemoveByMatchGraph(AllMatchesType& matches);

void SimilarityRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2, double trainThreshold, double testThreshold);

void FundamentalRansac(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2);

ImageProcessor::ImageProcessor(const std::string& configFile, const std::string& workDir) {
    workDir_ = workDir;
    configFile_ = configFile;
    pFrames_ = std::make_shared<std::vector<Frame>>();
    pMatches_ = std::make_shared<AllMatchesType>();
    std::string featureMethod = "sift";
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    if (fs.isOpened()) {
        featureMethod = fs["feature"]["method"].empty() ? std::string("sift") : (std::string)fs["feature"]["method"];
        fs.release();
    }
    if (featureMethod == "xfeat") {
        pExtractor_ = std::make_shared<XFeatExtractor>(configFile);
    } else if (featureMethod == "superpoint") {
        pExtractor_ = std::make_shared<SuperPointExtractor>(configFile);
    } else {
        pExtractor_ = std::make_shared<SiftExtractor>(configFile);
    }
    featureMethod_ = featureMethod;
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
    //ReadFrames(workDir_ + "/Frames.txt", pFrames_);
    //ReadMatches(workDir_ + "/Matches.txt", pMatches_);

    // Check if the frames and matches are already loaded and valid; if so, exit early.
    if (pFrames_ && pMatches_ && pFrames_->size() > 0 && pMatches_->size() > 0) {
        return;
    }

    // Collect paths of images in the specified directory with png and jpg extensions.
    std::vector<std::string> imgPaths;
    CollectImagePath(imgDir + "/", "png", imgPaths);
    CollectImagePath(imgDir + "/", "jpg", imgPaths);

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
        std::cout << "\rExtracting (" << idx << "/" << imgPaths.size() << "): " << path << "  " << std::flush;
        Frame frame(configFile_, workDir_);
        frame.SetFeatureExtractor(pExtractor_);
        bool flag = frame.LoadAndExtract(path);
        pFrames_->push_back(frame);
        if (frontendCallback_) {
            std::vector<cv::Point2f> kpts;
            kpts.reserve(frame.keypointList_.size());
            for (const auto& k : frame.keypointList_) {
                kpts.push_back(k.pt);
            }
            auto t1 = cv::getTickCount();
            double estCost = double(t1 - t0) / cv::getTickFrequency();
            extractCost_ = estCost;
            frontendCallback_((int)pFrames_->size() - 1, frame.image_, kpts, -1, cv::Mat(), {}, {},
                              (int)imgPaths.size(), (int)pFrames_->size(), 0, 0, estCost, 0.0);
        }
        auto t1 = cv::getTickCount();
        double fs = 1. / cv::getTickFrequency();
        std::cout << "Keypoints: " << frame.keypointList_.size() << "  Cost: " << (t1 - t0) * fs << " s    ";
    }
    std::cout << "\n";

    // Match all extracted frames to find correspondences between them.
    MatchAll();

    // Save the frames and matches back to files for future use.
    SaveFrames(workDir_ + "/Frames.txt", pFrames_);
    SaveMatches(workDir_ + "/Matches.txt", pMatches_);
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
        int width, height;
        in >> width >> height;
        frame.SetImageSize(cv::Size(width, height));
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
        out << frame.Width() << " " << frame.Height() << "\n";
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

#ifdef _WIN32
void CollectImagePath(const std::string& dirPath, std::string format, std::vector<std::string>& files) {
    intptr_t hFile = 0;
    struct _finddata_t fileInfo;
    std::string p;
    if ((hFile = _findfirst(p.assign(dirPath).append("\\*." + format).c_str(), &fileInfo)) != -1) {
        do {
            files.push_back(p.assign(dirPath).append("\\").append(fileInfo.name));
        } while (_findnext(hFile, &fileInfo) == 0);
        _findclose(hFile);
    }
}
#else
void CollectImagePath(const std::string& dirPath, std::string format, std::vector<std::string>& files) {
    DIR* dir = opendir(dirPath.c_str());
    if (!dir) {
        return;
    }

    struct dirent* entry;
    std::string pattern = "*." + format;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (fnmatch(pattern.c_str(), entry->d_name, 0) == 0) {
            files.push_back(dirPath + "/" + entry->d_name);
        }
    }
    closedir(dir);
}
#endif

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

    // Total number of pairs to be processed according to the match graph.
    int totalPairCnt = 0;
    for (int i = 0; i < frames.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (matchGraph.at<int>(i, j) != 0) {
                ++totalPairCnt;
            }
        }
    }

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
            std::cout << "\rMatching " << i << ", " << j << "  Total Matched: " << matchCnt << " pairs  Cost: " << tcost << " s   " << std::flush;
            std::vector<cv::DMatch> match_ij, tmpij;
            // Perform cross matching between descriptors of two frames.
            pExtractor_->Match(frames[i].descList_, frames[i].binaryDescs_,
                               frames[j].descList_, frames[j].binaryDescs_, match_ij);

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

            // Show the current frame's matches in the viewer.
            if (frontendCallback_) {
                std::vector<cv::Point2f> kptsI;
                kptsI.reserve(frames[i].keypointList_.size());
                for (const auto& k : frames[i].keypointList_) {
                    kptsI.push_back(k.pt);
                }
                std::vector<cv::Point2f> ptsI, ptsJ;
                ptsI.reserve(match_ij.size());
                ptsJ.reserve(match_ij.size());
                for (const auto& m : match_ij) {
                    ptsI.push_back(frames[i].keypointList_[m.queryIdx].pt);
                    ptsJ.push_back(frames[j].keypointList_[m.trainIdx].pt);
                }
                auto tm = cv::getTickCount();
                double matchCost = double(tm - t0) / cv::getTickFrequency();
                frontendCallback_(i, frames[i].image_, kptsI, j, frames[j].image_, ptsJ, ptsI,
                                  (int)frames.size(), i + 1, totalPairCnt, matchCnt,
                                  extractCost_, matchCost);
            }
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

    auto t0 = cv::getTickCount();

    // Initialize the match graph as a zero matrix of size equal to the number of frames.
    matchGraph = cv::Mat::zeros(frames.size(), frames.size(), CV_32S);
    for (int i = 0; i < frames.size(); ++i) {
        auto t1 = cv::getTickCount();
        double tcost = double(t1 - t0) / cv::getTickFrequency();
        std::cout << "\rBuilding match graph: " << i << "/" << frames.size() << "  Cost: " << tcost <<" s   " << std::flush;
        for (int j = 0; j < i; ++j) {
            std::vector<cv::DMatch> match_ij, low_match_ij;
            // Perform cross matching between low-resolution descriptors with a specific ratio.
            pExtractor_->MatchLowRes(frames[i].lDescList_, frames[j].lDescList_, low_match_ij);

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
                matchGraph.at<int>(i, j) = matchGraph.at<int>(j, i) = n;
            }
        }
    }

    // Determine the significance threshold for each row in the match graph.
    for (int i = 0; i < matchGraph.rows; ++i) {
        int maxCnt = 0;
        std::vector<int> cntList;
        for (int j = 0; j < matchGraph.cols; ++j) {
            maxCnt = std::max(maxCnt, matchGraph.at<int>(i, j));
            cntList.push_back(matchGraph.at<int>(i, j));
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
            if (matchGraph.at<int>(i, j) < cntThreshold) {
                matchGraph.at<int>(i, j) = 0;
            }
        }
    }

    // Ensure the match graph is symmetric.
    for (int i = 0; i < matchGraph.rows; ++i) {
        for (int j = 0; j < matchGraph.cols; ++j) {
            matchGraph.at<int>(i, j) = matchGraph.at<int>(j, i) =
                std::max(matchGraph.at<int>(i, j), matchGraph.at<int>(j, i));
        }
    }

    // Clear the match lists based on the updated match graph values.
    for (int i = 0; i < matches.size(); ++i) {
        for (int j = 0; j < matches[i].size(); ++j) {
            if (matches[i][j].size() > matchGraph.at<int>(i, j)) {
                matches[i][j].clear();
            }
        }
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
    if (matches.size() < 2) {
        matches.clear();
        return;
    }
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