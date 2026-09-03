#pragma once
#include <string>
#include <functional>
#include "Frame.h"
#include "FeatureExtractor/FeatureExtractor.h"

namespace sfm {
typedef std::vector<std::vector<std::vector<cv::DMatch>>> AllMatchesType;

/**
 * @brief Processes images for Structure from Motion (SfM) applications.
 *
 * This class encapsulates functionality for processing a set of images to extract features, match them across images,
 * and organize the resulting data for SfM reconstruction. It provides methods to read and save frame data and matches,
 * perform feature extraction and matching, and access processed data. The class relies on configuration and working directories
 * to manage resources and outputs.
 *
 * Functions:
 * - ExtractAndMatchAll: Coordinates the extraction of features from images and matches them across the image set.
 * - FramePtr: Provides access to the processed frames.
 * - MatchesPtr: Provides access to the matches found between frames.
 *
 * Detailed Description:
 * - The ImageProcessor class operates by loading images from a specified directory, processing each image to extract keypoints
 *   and descriptors, and then performing pairwise matching between images. Match results and keypoint data are stored
 *   in vectors and can be accessed via shared pointers.
 *
 * Usage:
 * - An instance of ImageProcessor is created with paths to a configuration file and a working directory.
 * - The ExtractAndMatchAll method is called with an image directory path to initiate processing.
 * - The results can be retrieved using FramePtr and MatchesPtr for further SfM processing or analysis.
 *
 * @param configFile Path to the configuration file that contains settings for feature extraction and matching.
 * @param workDir Path to the working directory where intermediate and final results are stored.
 */
class ImageProcessor {
public:
	ImageProcessor(const std::string& configFile, const std::string& workDir);
	void ExtractAndMatchAll(const std::string& imgDir);
	const std::shared_ptr<std::vector<Frame>>& FramePtr();
	const std::shared_ptr<AllMatchesType>& MatchesPtr();
	const std::string& Method() const { return featureMethod_; }

	// Callback invoked with the frontend visualization data:
	// current frame id, its image and feature points, and (during matching) a reference
	// frame id with the matched point pairs. A negative ref frame id means feature-only.
	// Progress info: total images / current image index (1-based), total pairs / current
	// pair index (1-based), feature-extraction time (cumulative during extraction,
	// total once finished) and matching time (cumulative), both in seconds.
	using FrontendCallback = std::function<void(int, const cv::Mat&, const std::vector<cv::Point2f>&,
	                                           int, const cv::Mat&, const std::vector<cv::Point2f>&,
	                                           const std::vector<cv::Point2f>&,
	                                           int, int, int, int, double, double)>;
	void SetFrontendCallback(const FrontendCallback& callback) { frontendCallback_ = callback; }

private:
	void MatchAll();
	void BuildMatchGraph(cv::Mat& matchGraph);
	void SaveFrames(const std::string& path, const std::shared_ptr<std::vector<Frame>>& pFrames);
    void ReadFrames(const std::string& path, std::shared_ptr<std::vector<Frame>>& pFrames);
    void SaveMatches(const std::string& path, const std::shared_ptr<AllMatchesType>& pMatches);
    void ReadMatches(const std::string& path, std::shared_ptr<AllMatchesType>& pMatches);

private:
	std::string configFile_;
	std::string workDir_;
	std::string featureMethod_;
	std::shared_ptr<std::vector<Frame>> pFrames_ = nullptr;
	std::shared_ptr<AllMatchesType> pMatches_ = nullptr;
	std::shared_ptr<AllMatchesType> pLowMatches_ = nullptr;
	std::shared_ptr<FeatureExtractor> pExtractor_ = nullptr;
	FrontendCallback frontendCallback_ = nullptr;
	double extractCost_ = 0.0;
};
}