#pragma once
#include <string>
#include <opencv2/opencv.hpp>

namespace sfm {
class Frame {
public:
	Frame() {}
	Frame(const std::string& configFile, const std::string& workDir);
	bool LoadAndExtract(const std::string& path);
	void SetPose(const cv::Mat& R_i_0, const cv::Mat& t_i_0);
	void SetIntrinsics(const double* intrinsicsPtr, const size_t intrinsicsSize);
	void SetImageSize(const cv::Size size);
	const cv::Mat& GetPose();
	const cv::Mat& RotationMatrix();
	const cv::Mat& TranslationVector();
	const cv::Mat& CameraMat();
	const int Width() const;
	const int Height() const;
	const int LowResWidth();
	const int LowResHeight();
	std::vector<int> GetKeypointsId();
	void UndistortPoints(const std::vector<cv::Point2f>& distPts, std::vector<cv::Point2f>& undistPts, cv::Mat newK = cv::Mat());
	void UndistortPoint(const cv::Point2f& distPt, cv::Point2f& undistPt);

private:
	bool ExtractAndDescript();

public:
	std::string imagePath_;
	cv::Mat image_;
	std::vector<cv::KeyPoint> keypointList_; // we save featureId to KeyPoint::class_id
	std::vector<cv::Vec3b> colorList_; // keypoints color
	cv::Mat descList_;
	std::vector<uint64_t> binaryDescs_;
	cv::Mat lImage_;
	std::vector<cv::KeyPoint> lKeyPts_;
	cv::Mat lDescList_;
	bool hasPose_ = false;
	cv::Mat R_i_w_;
	cv::Mat t_i_w_;
	cv::Mat T_i_w_;
	cv::Mat K_;
	cv::Mat D_;
	std::vector<double> intrinsics_;

private:
	std::string workDir_;
	int maxFeatureNum_ = -1;
	cv::Size imgSize_ = cv::Size(1500, 2000);
	cv::Size lowResSize_ = cv::Size(225, 300);
};

}