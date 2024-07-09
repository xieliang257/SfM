#include "Frame.h"
#include "opencv2/opencv.hpp"
#include "direct.h"

namespace sfm {
/**
* @brief Initializes a Frame object using settings from a configuration file.
*
* This constructor sets up a Frame by reading camera parameters and feature extraction settings from a specified
* configuration file. It loads the maximum number of features to detect, the camera's intrinsic matrix, and distortion
* coefficients which are essential for various computer vision tasks such as feature matching and 3D reconstruction.
*
* @param configFile Path to the configuration file containing settings for the frame.
* @param workDir Directory where the frame's output data will be stored.
*/
Frame::Frame(const std::string& configFile, const std::string& workDir) {
    workDir_ = workDir;
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open file: " << configFile << std::endl;
        return;
    }

    maxFeatureNum_ = fs["feature"]["feature_count"];
    double f = fs["camera"]["f"];
    double cx = fs["camera"]["cx"];
    double cy = fs["camera"]["cy"];
    K_ = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    intrinsics_.push_back(f);
    intrinsics_.push_back(cx);
    intrinsics_.push_back(cy);
    std::vector<double> dists;
    fs["camera"]["dist_coeffs"] >> dists;
    intrinsics_.insert(intrinsics_.end(), dists.begin(), dists.end());
    D_ = cv::Mat::zeros(5, 1, CV_64F);
    std::memcpy(D_.data, dists.data(), sizeof(double) * dists.size());
    fs.release();
}

void Frame::SetPose(const cv::Mat& R_i_0, const cv::Mat& t_i_0) {
    R_i_w_ = R_i_0.clone();
    t_i_w_ = t_i_0.clone();
    T_i_w_.create(3, 4, R_i_w_.type());
    R_i_w_.copyTo(T_i_w_(cv::Rect(0, 0, 3, 3)));
    t_i_w_.copyTo(T_i_w_(cv::Rect(3, 0, 1, 3)));
    hasPose_ = true;
}

/**
 * @brief Sets the camera's intrinsic parameters and updates the camera matrix and distortion coefficients.
 *
 * @param intrinsicsPtr Pointer to an array of doubles that includes the camera's intrinsic parameters.
 *                      The array should start with the focal length, principal point coordinates (cx, cy),
 *                      followed by any distortion coefficients.
 * @param intrinsicsSize The number of elements in the array pointed to by intrinsicsPtr.
 * 
 * Usage Example:
 * double intrinsics[] = {f, cx, cy, k1}; // Example parameters
 * frame.SetIntrinsics(intrinsics, 4); // Setting intrinsics and distortion coefficients
 */
void Frame::SetIntrinsics(const double* intrinsicsPtr, const size_t intrinsicsSize) {
    for (size_t i = 0; i < intrinsicsSize; ++i) {
        intrinsics_[i] = intrinsicsPtr[i];
    }
    K_.at<double>(0, 0) = intrinsicsPtr[0];
    K_.at<double>(1, 1) = intrinsicsPtr[0];
    K_.at<double>(0, 2) = intrinsicsPtr[1];
    K_.at<double>(1, 2) = intrinsicsPtr[2];
    for (size_t i = 0; i + 3 < intrinsicsSize; ++i) {
        D_.at<double>(i, 0) = intrinsicsPtr[i + 3];
    }
}

const int Frame::Width() {
    return imgSize_.width;
}

const int Frame::Height() {
    return imgSize_.height;
}

const int Frame::LowResWidth() {
    return lowResSize_.width;
}

const int Frame::LowResHeight() {
    return lowResSize_.height;
}

const cv::Mat& Frame::GetPose() {
    return T_i_w_;
}

const cv::Mat& Frame::CameraMat() {
    return K_;
}

const cv::Mat& Frame::RotationMatrix() {
    return R_i_w_;
}

const cv::Mat& Frame::TranslationVector() {
    return t_i_w_;
}

std::vector<int> Frame::GetKeypointsId() {
    std::vector<int> ids;
    for (auto& kpt : keypointList_) {
        ids.push_back(kpt.class_id);
    }
    return ids;
}

void Frame::UndistortPoints(const std::vector<cv::Point2f>& distPts, std::vector<cv::Point2f>& undistPts, cv::Mat newK) {
    cv::undistortPoints(distPts, undistPts, K_, D_, cv::Mat(), newK.empty() ? K_ : newK);
}

void Frame::UndistortPoint(const cv::Point2f& distPt, cv::Point2f& undistPt) {
    std::vector<cv::Point2f> distPts = { distPt }, undistPts;
    UndistortPoints(distPts, undistPts);
    undistPt = undistPts[0];
}

/**
 * @brief Extracts keypoints from an image using SIFT and distributes them uniformly across the image.
 *
 * This function performs feature detection using the SIFT algorithm configured to detect a high number of preliminary keypoints.
 * The keypoints are then sorted by their response value to prioritize stronger keypoints. To ensure uniform distribution, the
 * image is divided into blocks, and a fixed number of keypoints are selected from each block. If the number of selected keypoints
 * is less than the desired count, additional keypoints are added from those not initially selected to meet the required number.
 *
 * @param image The input image from which to extract features.
 * @param blocks The number of blocks the image is divided into for uniform keypoint distribution.
 * @param kptCnt The total number of keypoints desired.
 * @param quality Quality level for keypoint detection; influences the threshold for accepting keypoints.
 * @param kpts Output vector where the extracted keypoints are stored.
 */
void Extract(const cv::Mat& image, int blocks, int kptCnt, double quality, std::vector<cv::KeyPoint>& kpts) {
    auto sift = cv::SIFT::create(kptCnt * 10, 3, quality, 10);
    sift->detect(image, kpts);

    // Sort keypoints by response to prioritize stronger ones.
    std::sort(kpts.begin(), kpts.end(), [](cv::KeyPoint& k1, cv::KeyPoint& k2) {return k1.response > k2.response; });

    // Filters keypoints, marking a 3x3 area on a mask to ensure each keypoint is unique.
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

    // 3D vector to hold keypoints for each block.
    std::vector<std::vector<std::vector<cv::KeyPoint>>> blockKeypts;
    blockKeypts.resize(image.rows / w + 1);
    for (auto& b : blockKeypts) {
        b.resize(image.cols / w + 1);
    }

    // Hold additional keypoints if needed.
    std::vector<cv::KeyPoint> otherPts;

    for (auto& k : kpts) {
        int i = k.pt.y / w;
        int j = k.pt.x / w;

        // Ensure that each block contains no more than a proportional number of keypoints.
        if (blockKeypts[i][j].size() <= kptCnt / blocks) {
            blockKeypts[i][j].push_back(k);
        }
        else {
            otherPts.push_back(k);
        }
    }

    // Clear original keypoints vector to populate with uniformly distributed keypoints.
    kpts.clear();
    for (int i = 0; i < blockKeypts.size(); ++i) {
        for (int j = 0; j < blockKeypts[i].size(); ++j) {
            kpts.insert(kpts.end(), blockKeypts[i][j].begin(), blockKeypts[i][j].end());
        }
    }

    // Ensure the total number of keypoints meets the specified requirement.
    if (kpts.size() > kptCnt) {
        kpts.resize(kptCnt);
        return;
    }

    if (kpts.size() < kptCnt) {
        // Add additional keypoints from those not initially selected to meet the desired count.
        for (auto& other : otherPts) {
            kpts.push_back(other);
            if (kpts.size() >= kptCnt) {
                break;
            }
        }
    }
}

/**
 * @brief Converts floating-point descriptors to binary format for efficient matching.
 *
 * This function transforms SIFT descriptors from a floating-point representation to a binary format using a simple
 * thresholding method. Each descriptor's elements are compared to the average value of the descriptor; elements above
 * the average are encoded as 1, and those below as 0. This binary encoding is stored across two 64-bit integers per
 * descriptor, making it suitable for fast Hamming distance computation.
 *
 * @param descMat_ The input matrix of descriptors where each row is a descriptor in floating-point format.
 * @param binaryDesc Output vector where each descriptor's binary representation is stored as pairs of uint64_t.
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
 * @brief Extracts features from the frame's images and computes their descriptors.
 *
 * This method processes both the primary image and a lower-resolution version of it to extract keypoints
 * using the SIFT algorithm. It then calculates the descriptors for these keypoints and normalizes them. The method
 * also converts these descriptors to a binary format for efficient matching and records the color information at
 * each keypoint location for potential use in feature matching and tracking.
 *
 * @return True if keypoints are successfully extracted and processed; false otherwise.
 */
bool Frame::ExtractAndDescript() {
    cv::Mat grayImage, lGrayImage;
    cv::cvtColor(image_, grayImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(lImage_, lGrayImage, cv::COLOR_BGR2GRAY);
    auto sift = cv::SIFT::create(maxFeatureNum_);

    // Extract keypoints from both the main image and the lower-resolution image.
    Extract(grayImage, 100, maxFeatureNum_, 0.005, keypointList_);
    Extract(lGrayImage, 100, 300, 0.005, lKeyPts_);

    sift->compute(grayImage, keypointList_, descList_);
    sift->compute(lGrayImage, lKeyPts_, lDescList_);

    // Normalize the descriptors to unit length
    for (int i = 0; i < descList_.rows; ++i) {
        descList_.row(i) /= cv::norm(descList_.row(i));
    }
    for (int i = 0; i < lDescList_.rows; ++i) {
        lDescList_.row(i) /= cv::norm(lDescList_.row(i));
    }

    // Convert float descriptors to binary descriptors for efficient matching.
    ToBinaryDesc(descList_, binaryDescs_);

    colorList_.clear();
    for (const auto& kpt : keypointList_) {
        int x = kpt.pt.x;
        int y = kpt.pt.y;
        colorList_.push_back(image_.at<cv::Vec3b>(y, x));
    }

    if (keypointList_.size() == 0) {
        return false;
    }
    for (auto& key : keypointList_) {
        key.class_id = -1;
    }

    return true;
}

bool Frame::LoadAndExtract(const std::string& path) {
    imagePath_ = path;
    image_ = cv::imread(path);
    if (image_.empty()) {
        return false;
    }
    imgSize_ = image_.size();

    double scale = 300. / std::max(image_.cols, image_.rows);
    cv::resize(image_, lImage_, cv::Size(image_.cols*scale, image_.rows*scale), 0, 0, cv::INTER_AREA);
    lowResSize_ = lImage_.size();
    bool extractFlag = ExtractAndDescript();

    //image_.release();
    //lImage_.release();
    return extractFlag;
}
}