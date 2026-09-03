#include "Frame.h"
#include <algorithm>
#include <iostream>
#include "opencv2/opencv.hpp"

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

void Frame::SetImageSize(const cv::Size size) {
    imgSize_ = size;
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

const int Frame::Width() const {
    return imgSize_.width;
}

const int Frame::Height() const {
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
 * @brief Extracts features from the frame's images and computes their descriptors.
 *
 * This method processes both the primary image and a lower-resolution version of it to extract keypoints
 * using the configured feature backend. It then records the color information at each keypoint location for
 * potential use in feature matching and tracking.
 *
 * @return True if keypoints are successfully extracted and processed; false otherwise.
 */
bool Frame::ExtractAndDescript() {
    cv::Mat grayImage, lGrayImage;
    cv::cvtColor(image_, grayImage, cv::COLOR_BGR2GRAY);
    cv::cvtColor(lImage_, lGrayImage, cv::COLOR_BGR2GRAY);

    if (!pExtractor_) {
        std::cerr << "Frame: feature extractor not set, cannot extract features" << std::endl;
        return false;
    }

    std::vector<float> scores;
    if (!pExtractor_->Extract(grayImage, keypointList_, descList_, binaryDescs_, scores)) {
        return false;
    }
    pExtractor_->ExtractLowRes(lGrayImage, lKeyPts_, lDescList_);
    std::cout << "pts1, pts2: " << lKeyPts_.size()<<", " << keypointList_.size()<<"\n";
    colorList_.clear();
    for (const auto& kpt : keypointList_) {
        int x = std::min(std::max((int)kpt.pt.x, 0), image_.cols - 1);
        int y = std::min(std::max((int)kpt.pt.y, 0), image_.rows - 1);
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

    double scale = 600. / std::max(image_.cols, image_.rows);
    cv::resize(image_, lImage_, cv::Size(image_.cols*scale, image_.rows*scale), 0, 0, cv::INTER_AREA);
    lowResSize_ = lImage_.size();
    bool extractFlag = ExtractAndDescript();

    //image_.release();
    //lImage_.release();
    return extractFlag;
}
}