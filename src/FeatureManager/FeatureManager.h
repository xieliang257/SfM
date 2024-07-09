#pragma once
#include <opencv2/opencv.hpp>
#include "ImageProcessor/ImageProcessor.h"
#include "Structure.h"

namespace sfm {
class FeatureManager {
public:
	FeatureManager();
	void BuildSFMFeatureMap(const std::shared_ptr<std::vector<Frame>>& pFrames, const std::shared_ptr<AllMatchesType>& pMatches);
	const std::shared_ptr<std::map<size_t, SFMFeature>>& SFMFeaturesPtr();

private:
	void BuildFeatureMap();

private:
	std::shared_ptr<std::vector<Frame>> pFrames_ = nullptr;
	std::shared_ptr<AllMatchesType> pMatches_ = nullptr;

	// <featureId, msfFeature>
	std::shared_ptr<std::map<size_t, SFMFeature>> pSfmFeatures_ = nullptr;
};
}