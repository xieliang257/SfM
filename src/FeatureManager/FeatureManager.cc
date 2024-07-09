#include "FeatureManager/FeatureManager.h"

namespace sfm {
FeatureManager::FeatureManager(){
	pSfmFeatures_ = std::make_shared<std::map<size_t, SFMFeature>>();
}

const std::shared_ptr<std::map<size_t, SFMFeature>>& FeatureManager::SFMFeaturesPtr() {
	return pSfmFeatures_;
}

/**
 * Constructs a feature map by assigning unique IDs to matched keypoints across multiple frames.
 * This mapping is crucial for subsequent 3D reconstruction tasks in Structure from Motion (SfM) workflows.
 */
void FeatureManager::BuildFeatureMap(){
	AllMatchesType& matches = *pMatches_;
	std::vector<Frame>& frames = *pFrames_;
	std::map<size_t, SFMFeature>& sfmFeatures = *pSfmFeatures_;

	// Initialize feature IDs for all keypoints in all frames to -1 (unassigned)
	std::vector<std::vector<int>> ids(frames.size());
	for (int i = 0; i < ids.size(); ++i) {
		ids[i].resize(frames[i].keypointList_.size(), -1);
		for (auto& id : ids[i]) {
			id = -1;
		}
	}

	// Generate unique feature IDs for matched keypoints
	int featureId = 0;
	for (int i = 0; i < matches.size(); ++i) {
		for (int j = 0; j < matches[i].size(); ++j) {
			// Skip if fewer than 30 matches to ensure robustness
			if (matches[i][j].size() < 30) {
				continue;
			}
			for (int k = 0; k < matches[i][j].size(); ++k) {
				int ptId_i = matches[i][j][k].queryIdx;
				int ptId_j = matches[i][j][k].trainIdx;
				// Assign new IDs or propagate existing ones
				if (ids[i][ptId_i] == -1 && ids[j][ptId_j] == -1) {
					ids[i][ptId_i] = featureId;
					ids[j][ptId_j] = featureId;
					featureId++;
				}
				else if (ids[i][ptId_i] == -1) {
					ids[i][ptId_i] = ids[j][ptId_j];
				}
				else if (ids[j][ptId_j] == -1) {
					ids[j][ptId_j] = ids[i][ptId_i];
				}
				else if (ids[i][ptId_i] != ids[j][ptId_j]) {
					// Resolve ID conflicts
					ids[i][ptId_i] = -1;
					ids[j][ptId_j] = -1;
				}
			}
		}
	}

	// Remove duplicate IDs within the same frame
	for (int i = 0; i < ids.size(); ++i) {
		for (int j = 0; j < ids[i].size(); ++j) {
			if (ids[i][j] == -1) {
				continue;
			}
			for (int k = j+1; k < ids[i].size(); ++k) {
				if (ids[i][j] == ids[i][k]) {
					// Mark duplicates as unassigned
					ids[i][k] = -1;
				}
			}
		}
	}

	// Populate the SFM features map
	for (int i = 0; i < ids.size(); ++i) {
		for (int j = 0; j < ids[i].size(); ++j) {
			if (ids[i][j] == -1) {
				continue;
			}
			int pointId = ids[i][j];
			cv::Point2f pt = frames[i].keypointList_[j].pt;
			cv::Vec3b color = frames[i].colorList_[j];

			// Create or update the feature in the global map
			if (sfmFeatures.find(pointId) == sfmFeatures.end()) {
				sfmFeatures.insert(std::make_pair(pointId, SFMFeature()));
			}
			Feature fea;
			fea.featureId = pointId;
			fea.imageId = i;
			fea.pt = pt;
			fea.color = color;
			sfmFeatures[pointId].features.push_back(fea);
		}
	}

	// Update keypoints with their final feature IDs
	for (int i = 0; i < frames.size(); ++i) {
		for (int j = 0; j < frames[i].keypointList_.size(); ++j) {
			frames[i].keypointList_[j].class_id = ids[i][j];
		}
	}
}

void FeatureManager::BuildSFMFeatureMap(const std::shared_ptr<std::vector<Frame>>& pFrames, const std::shared_ptr<AllMatchesType>& pMatches) {
	pFrames_ = pFrames;
	pMatches_ = pMatches;
	std::map<size_t, SFMFeature>& sfmFeatures = *pSfmFeatures_;

	BuildFeatureMap();
}
}