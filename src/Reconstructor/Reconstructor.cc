#include <ceres/ceres.h>
#include "Reconstructor/Reconstructor.h"
#include "Reconstructor/PoseSolver.h"
#include "Reconstructor/BaCostFunc.h"

namespace sfm {
Reconstructor::Reconstructor(const std::string& configFile, const std::string& workDir) {
	featureManager_ = std::make_shared<FeatureManager>();
	workDir_ = workDir;
	cv::FileStorage fs(configFile, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Failed to open file: " << configFile << std::endl;
	}

	intrinsicsSize_ = 3;
	std::vector<double> dists;
	fs["camera"]["dist_coeffs"] >> dists;
	intrinsicsSize_ += dists.size();

	fs["optimize_intrinsic_params"] >> optimizeIntrinsicParams_;
	fs["share_intrinsic_params"] >> shareIntrinsicParams_;
	fs.release();
}

/**
 * @brief Reconstruct the structure from motion (SfM) using input frames and matches.
 *
 * This function initializes the reconstruction process, iterates through each frame incrementally,
 * and performs global bundle adjustment periodically to optimize the camera poses and structure.
 *
 * @param pFrames A shared pointer to a vector of Frame objects, which represent the input images and their properties.
 * @param pMatches A shared pointer to AllMatchesType, which encapsulates all the feature matches between frames.
 */
void Reconstructor::Reconstruct(const std::shared_ptr<std::vector<Frame>>& pFrames,
	const std::shared_ptr<AllMatchesType>& pMatches) {
	pFrames_ = pFrames;
	pMatches_ = pMatches;
	std::cout << "Total frames: " << pFrames_->size() << "\n\n";
	auto t0 = cv::getTickCount();

	// Run feature management to prepare for SFM.
	featureManager_->BuildSFMFeatureMap(pFrames_, pMatches_);
	pSfmFeatures_ = featureManager_->SFMFeaturesPtr();

	// Attempt to start the reconstruction; exit if not successful.
	if (!Start()) {
		return;
	}

	int solvedFrames = 2;
	while (1) {
		// Get the next frame to process based on current progress.
		int incrementId = Increment();

		// Exit the loop if no more frames need processing.
		if (incrementId == -1) {
			break;
		}
		++solvedFrames;

		// Perform global bundle adjustment every 10 frames.
		if (solvedFrames % 10 == 0) {
			GlobalBA();
		}

		// Retrieve keypoints' IDs for the current frame and perform checks and re-triangulation.
		std::vector<int> ids = (*pFrames)[incrementId].GetKeypointsId();

		double reprojErrThr = 3., cosAngThr = cos(1 * CV_PI / 180);
		CheckAndRetrangulation(ids, reprojErrThr, cosAngThr);
		auto t1 = cv::getTickCount();
		double tcost = double(t1 - t0)/cv::getTickFrequency();
		std::cout << "Total " << solvedFrames << " images solved.  Cost: " << tcost << " s    ";
	}

	// Final global bundle adjustment after all frames are processed.
	GlobalBA();

	// Perform a final check and re-triangulation for all features.
	double reprojErrThr = 1., cosAngThr = cos(3 * CV_PI / 180);
	CheckAndRetrangulation(reprojErrThr, cosAngThr);
	std::cout << "\nReconstruction done\n\n";
}

/**
 * @brief Initializes the structure from motion (SfM) reconstruction process by finding and setting the initial pair of frames.
 *
 * This function attempts to find the first pair of frames that will serve as the base for the entire reconstruction process.
 * If the first pair is found successfully, it sets their initial poses and attempts to triangulate initial features.
 *
 * @return Returns true if the initial frame pair is successfully found and initialized; otherwise, returns false.
 */
bool Reconstructor::Start() {
	// Initialize frame IDs to invalid state
	startPair_.first = startPair_.second = -1;

	// Matrices for rotation and translation from frame1 to frame2
	cv::Mat R_2_1, t_2_1;

	// Try to find the first pair of frames and their relative pose
	if (!FindFirstPair(startPair_.first, startPair_.second, R_2_1, t_2_1)) {
		std::cout << "SfM start failed\n";
		return false;
	}

	auto& frames = *pFrames_;
	std::cout << "SfM start with:\n"
		<< "Image 1: " << frames[startPair_.first].imagePath_ << "\n"
		<< "Image 2: " << frames[startPair_.second].imagePath_ << "\n\n";

	// Set the pose of the first frame as the world origin (identity matrix for rotation and zero vector for translation)
	frames[startPair_.first].SetPose(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));

	// Set the pose of the second frame relative to the first frame using the found rotation and translation
	frames[startPair_.second].SetPose(R_2_1, t_2_1);

	auto& sfmFeatures = *pSfmFeatures_;

	// Define a cosine angle threshold for feature triangulation
	double cosAngleThreahold = cos(1 * CV_PI / 180);

	// Triangulate features from the first frame
	for (auto& kpt : frames[startPair_.first].keypointList_) {
		sfmFeatures[kpt.class_id].Triangulate(pFrames_, 5, cosAngleThreahold);
	}
	return true;
}

/**
 * @brief Finds the best initial pair of frames for starting the SfM.
 *
 * This function identifies the pair of frames with the most robust matches to serve as the base
 * for the SfM process. It evaluates the candidate frames based on the number of feature matches and
 * computes the initial relative pose between the selected pair using essential matrix decomposition.
 *
 * @param imageId1 ID of the first selected frame.
 * @param imageId2 ID of the second selected frame.
 * @param R Reference to a cv::Mat to store the rotation matrix between the two frames.
 * @param t Reference to a cv::Mat to store the translation vector between the two frames.
 * @return Returns true if a suitable pair is found and their initial relative pose is successfully computed;
 *         otherwise, returns false.
 */
bool Reconstructor::FindFirstPair(int& imageId1, int& imageId2, cv::Mat& R, cv::Mat& t) {
	// Initialize frame IDs to invalid values
	imageId1 = imageId2 = -1;

	const AllMatchesType& matches = *pMatches_;
	std::vector<Frame>& frames = *pFrames_;
	std::vector<int> candFramesId;

	// Loop to identify candidate frames based on the number of matches
	while (1) {
		int frameId = -1;
		size_t maxCnt = 0;
		for (int i = 0; i < matches.size(); ++i) {
			for (int j = 0; j < matches[i].size(); ++j) {
				if (maxCnt < matches[i][j].size()) {
					bool inCand = false;
					for (auto candId : candFramesId) {
						if (i == candId || j == candId) {
							inCand = true;
							break;
						}
					}
					if (!inCand) {
						frameId = i;
						maxCnt = matches[i][j].size();
					}
				}
			}
		}

		// Break if no candidates are found
		if (maxCnt == 0 || frameId == -1) {
			break;
		}

		// Add the frame ID to candidate list
		candFramesId.push_back(frameId);

		// Limit the number of candidates to 
		if (candFramesId.size() >= 10) {
			break;
		}
	}

	// Evaluate all candidate pairs to select the best based on inlier matches
	std::vector<cv::Vec2i> candidatePairs;
	for (auto candId : candFramesId) {
		for (int i = 0; i < matches.size(); ++i) {
			size_t maxCnt = 0;
			for (int j = 0; j < matches.size(); ++j) {
				maxCnt = std::max(std::max(matches[i][candId].size(), matches[candId][i].size()), maxCnt);
			}
			if (matches[i][candId].size() > maxCnt * 0.6) {
				candidatePairs.push_back(cv::Vec2i(i, candId));
			}
			if (matches[candId][i].size() > maxCnt * 0.6) {
				candidatePairs.push_back(cv::Vec2i(candId, i));
			}
		}
	}

	// Variable to track the highest ratio of inliers
	double maxInlierRatio = 0;

	// Loop through each candidate pair to compute the relative pose using the essential matrix
	for (int i = 0; i < candidatePairs.size(); ++i) {
		int frameId1 = candidatePairs[i](0);
		int frameId2 = candidatePairs[i](1);
		const std::vector<cv::DMatch>& match = matches[frameId1][frameId2];

		// Extract points from the matches
		std::vector<cv::Point2f> pts1, pts2;
		for (const auto& m : match) {
			int id1 = m.queryIdx;
			int id2 = m.trainIdx;
			pts1.push_back(frames[frameId1].keypointList_[id1].pt);
			pts2.push_back(frames[frameId2].keypointList_[id2].pt);
		}

		// Ensure sufficient points for reliable estimation
		if (pts1.size() < 50) {
			continue;
		}

		// Undistort points
		std::vector<cv::Point2f> undistoredPts1, undistoredPts2;
		frames[frameId1].UndistortPoints(pts1, undistoredPts1);
		frames[frameId2].UndistortPoints(pts2, undistoredPts2, frames[frameId1].K_);

		// Matrices for the relative pose
		cv::Mat R_2_1, t_2_1;

		// Triangulation results
		std::vector<TriangulateResult> points;
		PoseSolver::EssentialSolver(frames[frameId1].K_, undistoredPts1, undistoredPts2, R_2_1, t_2_1, points, 3);

		// Count inliers based on triangulation status and angle view
		int inlierCnt = 0;
		for (auto p : points) {
			if (p.statu && p.cosAngleView < cos(5 * CV_PI / 180)) {
				inlierCnt++;
			}
		}

		// Calculate the ratio of inliers and update the best pair if the current one has a higher inlier ratio
		double ratio = inlierCnt / double(undistoredPts1.size());
		if (ratio > maxInlierRatio) {
			maxInlierRatio = ratio;
			imageId1 = frameId1;
			imageId2 = frameId2;
			R_2_1.copyTo(R);
			t_2_1.copyTo(t);
		}

		if (inlierCnt == points.size()) {
			break;
		}
	}

	// Return false if no suitable pair is found, otherwise return true
	if (imageId1 == -1 || imageId2 == -1) {
		return false;
	}
	return true;
}

/**
 * @brief Processes the next frame in the sequence, attempting to estimate its pose and triangulate new points.
 *
 * This function iteratively selects the next frame based on its potential to add valuable information
 * to the reconstruction (based on the number of visible features). It attempts to estimate the pose
 * using PnP (Perspective-n-Point) and Essential Homography methods and, if successful, updates the
 * frame's pose and triangulates new points.
 *
 * @return Returns the ID of the frame that was successfully processed, or -1 if no suitable frame can be found.
 */
int Reconstructor::Increment() {
	std::vector<Frame>& frames = *pFrames_;
	auto& sfmFeatures = *pSfmFeatures_;

	// Variable to store the selected frame ID
	int frameId = -1;
	static int pnpCnt = 0, ehCnt = 0;

	// Loop to find candidate frames based on the number of views (visible features)
	while (1) {
		// Populate viewsList_ if it's empty
		if (viewsList_.empty()) {
			// Track the maximum number of views
			int maxViews = -1;
			for (int i = 0; i < frames.size(); ++i) {
				int views = 0;

				// Only consider frames without an estimated pose
				if (!frames[i].hasPose_) {
					for (const auto& kpt : frames[i].keypointList_) {
						if (sfmFeatures[kpt.class_id].hasObject) {
							++views;
						}
					}
				}
				if (views > 0) {
					viewsList_.push_back(std::pair<int, int>(i, views));
					maxViews = std::max(maxViews, views);
				}
			}

			// Filter and sort candidate frames based on views
			if (maxViews > 0) {
				int id = 0;
				for (int i = 0; i < viewsList_.size(); ++i) {
					if (viewsList_[i].second >= maxViews * 0.8) {
						viewsList_[id] = viewsList_[i];
						++id;
					}
				}
				viewsList_.resize(id);
				std::sort(viewsList_.begin(), viewsList_.end(), [](std::pair<int, int> a, std::pair<int, int> b) {return a.second < b.second; });
				if (viewsList_.size() > 3) {
					viewsList_.assign(viewsList_.end() - 3, viewsList_.end());
				}
			}
		}

		// Exit if no candidate frames are available
		if (viewsList_.empty()) {
			return -1;
		}

		// Select the frame with the most views for processing
		frameId = viewsList_.back().first;
		viewsList_.pop_back();

		// Attempt to estimate the pose using PnP and Essential Homography
		cv::Mat Rpnp_i_w, Tpnp_i_w;
		cv::Mat Reh_i_w, Teh_i_w;
		bool pnpFlag = PnPPose(frameId, Rpnp_i_w, Tpnp_i_w);
		bool ehFlag = EHPose(frameId, Reh_i_w, Teh_i_w);
		if (pnpFlag) {
			frames[frameId].SetPose(Rpnp_i_w, Tpnp_i_w);
			++pnpCnt;
			break;
		}
		else if (ehFlag) {
			frames[frameId].SetPose(Reh_i_w, Teh_i_w);
			++ehCnt;
			break;
		}

		// Return -1 if no more frames can be processed
		if (viewsList_.empty()) {
			return -1;
		}
	}
	std::cout << "\rIncrease by PnP (" << pnpCnt << ") E or H (" << ehCnt << ")  ";


	// Triangulate new features for the selected frame
	double reprojErrThreshold = 3;
	double cosAngleThreshold = cos(3 * CV_PI / 180);
	for (const auto& kpt : frames[frameId].keypointList_) {
		int featureId = kpt.class_id;
		if (featureId != -1 && !sfmFeatures[featureId].hasObject) {
			sfmFeatures[featureId].Triangulate(pFrames_, reprojErrThreshold, cosAngleThreshold);
		}
	}

	// Return the ID of the processed frame
	return frameId;
}

/**
 * @brief Estimates the pose of a frame using the Perspective-n-Point (PnP) algorithm.
 *
 * This function attempts to estimate the camera pose by finding correspondences between 3D points in the world
 * and 2D points in the image frame. It requires a sufficient number of matches to compute a reliable pose.
 *
 * @param frameId The ID of the frame for which to estimate the pose.
 * @param R_i_w Reference to a cv::Mat that will hold the estimated rotation matrix.
 * @param t_i_w Reference to a cv::Mat that will hold the estimated translation vector.
 * @return Returns true if the pose was successfully estimated; otherwise, returns false.
 */
bool Reconstructor::PnPPose(int frameId, cv::Mat& R_i_w, cv::Mat& t_i_w) {
	std::vector<Frame>& frames = *pFrames_;
	AllMatchesType matches = *pMatches_;
	auto& sfmFeatures = *pSfmFeatures_;

	// Find 3D-2D pairs for PnP solver
	std::vector<cv::Point3f> objects;
	std::vector<cv::Point2f> observes;
	for (const auto& feature : frames[frameId].keypointList_) {
		int featureId = feature.class_id;
		if (sfmFeatures[featureId].hasObject) {
			objects.push_back(sfmFeatures[featureId].Xw);
			observes.push_back(feature.pt);
		}
	}

	// Check if there are enough points to perform PnP
	if (objects.size() < 30) {
		return false;
	}

	// Compute centroid and variance of observed points to assess distribution
	cv::Point2f sumpt(0.f, 0.f);
	for (auto p : observes) {
		sumpt += p;
	}
	sumpt *= 1. / observes.size();
	cv::Mat M = cv::Mat::zeros(2, 2, CV_32F);
	for (auto p : observes) {
		cv::Mat v(2, 1, CV_32F);
		v.at<float>(0, 0) = p.x - sumpt.x;
		v.at<float>(1, 0) = p.y - sumpt.y;
		M += v * v.t();
	}
	M /= objects.size();
	std::vector<float> values;
	cv::eigen(M, values);

	// Ensure points are well distributed
	if (sqrt(values[0] * values[1]) < 150 * 150) {
		return false;
	}

	// Undistort observed points
	std::vector<cv::Point2f> undistoredObserves;
	frames[frameId].UndistortPoints(observes, undistoredObserves);
	observes.clear();

	// Solve the PnP problem
	bool solveFlag = PoseSolver::PnPSolver(objects, undistoredObserves, frames[frameId].K_, R_i_w, t_i_w);
	return solveFlag;
}

/**
 * @brief Estimates the pose of a frame using Essential or Homography (EH) methods.
 *
 * This function estimates the camera pose by finding the best reference frame with sufficient matches.
 * It then attempts to estimate the pose using both the Essential Matrix and Homography approaches,
 * returning the result of the successful method.
 *
 * @param frameId The ID of the frame for which to estimate the pose.
 * @param R_i_w Reference to a cv::Mat that will hold the estimated rotation matrix.
 * @param t_i_w Reference to a cv::Mat that will hold the estimated translation vector.
 * @return Returns true if the pose was successfully estimated by either method; otherwise, returns false.
 */
bool Reconstructor::EHPose(int frameId, cv::Mat& R_i_w, cv::Mat& t_i_w) {
	std::vector<Frame>& frames = *pFrames_;
	AllMatchesType matches = *pMatches_;
	auto& sfmFeatures = *pSfmFeatures_;

	// Find a reference frame with the highest number of matches
	int refId = -1;
	int maxCnt = 0;
	for (int i = 0; i < frames.size(); ++i) {
		if (i != frameId && frames[i].hasPose_) {
			int cnt = std::max(matches[frameId][i].size(), matches[i][frameId].size());
			if (maxCnt < cnt) {
				maxCnt = cnt;
				refId = i;
			}
		}
	}

	// Return false if no suitable reference frame is found
	if (refId == -1) {
		return false;
	}

	// Prepare 2D-2D correspondence data
	// Points in the reference and current frames
	std::vector<cv::Point2f> pts1, pts2; 
	// Depth information for the reference frame points
	std::vector<double> depths1;         
	for (const auto& fea1 : frames[refId].keypointList_) {
		if (fea1.class_id == -1) {
			continue;
		}
		for (const auto& fea2 : frames[frameId].keypointList_) {
			if (fea2.class_id == fea1.class_id) {
				pts1.push_back(fea1.pt);
				pts2.push_back(fea2.pt);
				// Default depth
				double d1 = -1;

				// Calculate depth
				if (sfmFeatures[fea1.class_id].hasObject) {
					cv::Point3f Xw = sfmFeatures[fea1.class_id].Xw;
					cv::Mat Pw = (cv::Mat_<double>(3, 1) << Xw.x, Xw.y, Xw.z);
					cv::Mat Pcurr = frames[refId].RotationMatrix() * Pw + frames[refId].TranslationVector();
					d1 = Pcurr.at<double>(2, 0);
				}
				depths1.push_back(d1);
			}
		}
	}

	// Ensure there are enough correspondences
	if (depths1.size() < 30) {
		return false;
	}

	// Undistort the points
	std::vector<cv::Point2f> undistoredPts1, undistoredPts2;
	frames[refId].UndistortPoints(pts1, undistoredPts1);
	frames[frameId].UndistortPoints(pts2, undistoredPts2, frames[refId].K_);
	pts1.clear();
	pts2.clear();

	// Attempt to solve using the Essential Matrix
	// Rotation and translation from reference to current
	cv::Mat R_curr_ref, t_curr_ref;
	// Results of triangulation
	std::vector<TriangulateResult> points;
	bool essentialSolveFlag = PoseSolver::EssentialSolver(frames[refId].K_, depths1, undistoredPts1, undistoredPts2, R_curr_ref, t_curr_ref, points, 3);
	if (essentialSolveFlag) {
		R_i_w = R_curr_ref * frames[refId].RotationMatrix();
		t_i_w = R_curr_ref * frames[refId].TranslationVector() + t_curr_ref;
		return true;
	}

	// Attempt to solve using Homography
	// Homography rotation and translation
	cv::Mat Rh_curr_ref, th_curr_ref;
	bool homographySolveFlag = PoseSolver::HomographySolver(frames[refId].K_, depths1, undistoredPts1, undistoredPts2, Rh_curr_ref, th_curr_ref, 3);
	if (homographySolveFlag) {
		R_i_w = Rh_curr_ref * frames[refId].RotationMatrix();
		t_i_w = Rh_curr_ref * frames[refId].TranslationVector() + th_curr_ref;
		return true;
	}

	// Return false if both methods fail to estimate the pose
	return false;
}

/**
 * @brief Performs local bundle adjustment on the reconstruction.
 *
 * This function performs a local bundle adjustment to optimize the camera poses and structure
 * only for frames that are in close proximity to the newly added frame specified by incrementId.
 * The local adjustment helps to refine the reconstruction incrementally without the computational
 * cost of adjusting all frames.
 *
 * @param incrementId The index of the newly added frame which triggered the local bundle adjustment.
 */
void Reconstructor::LocalBA(int incrementId) {
	// List to hold the IDs of frames to be adjusted
	std::vector<int> constantFrameIds;

	// Loop through all frames to identify which ones have a pose and are not the newly added frame
	for (int i = 0; i < pFrames_->size(); ++i) {
		if ((*pFrames_)[i].hasPose_ && i != incrementId) {
			constantFrameIds.push_back(i);
		}
	}

	// Call the bundle adjustment implementation with the selected frame IDs
	BAImplement(constantFrameIds);
}

/**
 * @brief Performs global bundle adjustment across all frames in the reconstruction.
 *
 * Unlike LocalBA, GlobalBA optimizes the entire structure and camera parameters throughout
 * the entire dataset to ensure overall consistency and accuracy. It includes the first and
 * second frames as constants to anchor the global solution and minimize drift.
 */
void Reconstructor::GlobalBA() {
	std::vector<int> constantFrameIds = { startPair_.first, startPair_.second };
	BAImplement(constantFrameIds);
}

/**
 * @brief Implements the bundle adjustment process for optimizing frame poses and feature positions.
 *
 * This function sets up and solves a bundle adjustment problem using the Ceres solver. It optimizes the 3D points (features),
 * camera poses, and optionally intrinsic parameters of the cameras, based on the provided set of constant frame IDs.
 * The function supports shared and unique camera intrinsics across frames and implements several conditioning constraints
 * on the parameters like fixed principal points.
 *
 * @param constantFrameIds Vector of frame IDs that should remain constant during the adjustment, typically including anchor frames.
 */
void Reconstructor::BAImplement(const std::vector<int>& constantFrameIds) {
	// Caches for object points, camera poses, and intrinsics
	std::vector<double> objCache(pSfmFeatures_->size() * 3);
	std::vector<double> poseCache(pFrames_->size() * 6);
	size_t intrinsicBlockSize = intrinsicsSize_;
	if (!shareIntrinsicParams_){
		intrinsicBlockSize = intrinsicsSize_ * pFrames_->size();
	}
		
	std::vector<double> intrinsicsCache(intrinsicBlockSize);

	// Direct pointers to the cache data
	double* pObjCache = (double*)objCache.data();
	double* pPoseCache = (double*)poseCache.data();
	double* pIntrinsicsCache = (double*)intrinsicsCache.data();

	// Initialize intrinsics from the first frame if shared across all frames
	if (shareIntrinsicParams_) {
		auto& firstFrame = (*pFrames_)[0];
		std::vector<double> intrinsics = firstFrame.intrinsics_;
		for (size_t i = 0; i < intrinsics.size(); ++i) {
			intrinsicsCache[i] = intrinsics[i];
		}
	}

	// Maps to link feature IDs and frame IDs to their corresponding data blocks in caches
	std::map<size_t, double*> objMap;
	std::map<size_t, double*> transformationMap;
	std::map<size_t, double*> intrinsicsMap;
	ceres::Problem problem;

	// Conditionally add intrinsics to the problem
	if (shareIntrinsicParams_) {
		problem.AddParameterBlock(pIntrinsicsCache, intrinsicsSize_);
		if (!optimizeIntrinsicParams_) {
			problem.SetParameterBlockConstant(pIntrinsicsCache);
		}
		else {
			const std::vector<int> constantIntrinsicVec = { 1,2 };
			ceres::SubsetParameterization* subset_parameterization =
				new ceres::SubsetParameterization(intrinsicsSize_, constantIntrinsicVec);
			problem.SetParameterization(pIntrinsicsCache, subset_parameterization);
		}
	}

	// Iterate over all SFM features to prepare the optimization problem
	for (const auto& sfmFeaturePair : *pSfmFeatures_) {
		size_t featureId = sfmFeaturePair.first;
		const SFMFeature& sfmFeature = sfmFeaturePair.second;
		if (!sfmFeature.hasObject) {
			continue;
		}
		double* objData = nullptr;
		pObjCache[0] = (double)sfmFeature.Xw.x;
		pObjCache[1] = (double)sfmFeature.Xw.y;
		pObjCache[2] = (double)sfmFeature.Xw.z;
		if (objMap.find(featureId) == objMap.end()) {
			objMap.insert(std::make_pair(featureId, pObjCache));
			objData = pObjCache;
			problem.AddParameterBlock(objData, 3);
			pObjCache += 3;
		}

		// Link features to their frames and prepare the data blocks for pose and intrinsics
		for (const auto& feature : sfmFeature.features) {
			int frameId = feature.imageId;
			if (frameId == -1) {
				continue;
			}
			const Frame& curframe = (*pFrames_)[frameId];
			if (!curframe.hasPose_) {
				continue;
			}
			double* transData = nullptr;
			if (transformationMap.find(frameId) == transformationMap.end()) {
				cv::Mat angleAxis;
				cv::Rodrigues(curframe.R_i_w_, angleAxis);
				pPoseCache[0] = ((const double*)angleAxis.data)[0];
				pPoseCache[1] = ((const double*)angleAxis.data)[1];
				pPoseCache[2] = ((const double*)angleAxis.data)[2];
				const cv::Mat& trans = curframe.t_i_w_;
				pPoseCache[3] = ((const double*)trans.data)[0];
				pPoseCache[4] = ((const double*)trans.data)[1];
				pPoseCache[5] = ((const double*)trans.data)[2];
				transformationMap.insert(std::make_pair(frameId, pPoseCache));
				transData = pPoseCache;
				problem.AddParameterBlock(transData, 6);

				bool inConstant = false;
				for (const auto& constId : constantFrameIds) {
					if (constId == frameId) {
						inConstant = true;
						break;
					}
				}
				if (inConstant) {
					problem.SetParameterBlockConstant(transData);
				}
				pPoseCache += 6;
			}
			else {
				transData = transformationMap[frameId];
			}
			double* intrinsicsData = nullptr;
			if (shareIntrinsicParams_) {
				intrinsicsData = pIntrinsicsCache;
			}
			else {
				if (intrinsicsMap.find(frameId) == intrinsicsMap.end()) {
					const std::vector<double>& instrinsic = curframe.intrinsics_;
					for (size_t i = 0; i < instrinsic.size(); ++i) {
						pIntrinsicsCache[i] = instrinsic[i];
					}
					intrinsicsMap.insert(std::make_pair(frameId, pIntrinsicsCache));
					intrinsicsData = pIntrinsicsCache;
					problem.AddParameterBlock(intrinsicsData, intrinsicsSize_);
					if (!optimizeIntrinsicParams_) {
						problem.SetParameterBlockConstant(intrinsicsData);
					}
					else {
						const std::vector<int> constantIntrinsicVec = { 1,2 };
						ceres::SubsetParameterization* subset_parameterization =
							new ceres::SubsetParameterization(intrinsicsSize_, constantIntrinsicVec);
						problem.SetParameterization(intrinsicsData, subset_parameterization);
					}
					pIntrinsicsCache += intrinsicsSize_;
				}
				else {
					intrinsicsData = intrinsicsMap[frameId];
				}
			}

			// Create cost function for each observed point
			double px = (double)feature.pt.x;
			double py = (double)feature.pt.y;
			ceres::CostFunction* cost_function = CreateCostFunction(intrinsicsSize_, px, py);
			if (cost_function) {
				problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.5), intrinsicsData, transData, objData);
			}	
		}
	}

	// Configure and run the solver
	ceres::Solver::Options opt;
	opt.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
	opt.max_num_iterations = 10;
	opt.parameter_tolerance = 1e-6;
	opt.linear_solver_type = ceres::DENSE_SCHUR;
	opt.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(opt, &problem, &summary);
	//std::cout << summary.BriefReport() << std::endl;

	// Update the poses and intrinsics based on the optimized results
	for (auto& transformation : transformationMap) {
		size_t frameId = transformation.first;
		double* transData = (double*)transformation.second;
		cv::Mat rod(3, 1, CV_64F, transData), R;
		cv::Rodrigues(rod, R);
		cv::Mat t(3, 1, CV_64F, transData + 3);
		(*pFrames_)[frameId].SetPose(R, t);
	}

	// Update intrinsics
	if (optimizeIntrinsicParams_) {
		if (shareIntrinsicParams_) {
			for (auto& frame : *pFrames_) {
				frame.SetIntrinsics(pIntrinsicsCache, intrinsicsSize_);
			}
		}
		else {
			for (auto& intrinsics : intrinsicsMap) {
				size_t frameId = intrinsics.first;
				double* intrinsicsData = (double*)intrinsics.second;
				auto& frame = (*pFrames_)[frameId];
				frame.SetIntrinsics(intrinsicsData, intrinsicsSize_);
			}
		}
	}

	// Update the 3D points
	for (auto& obj : objMap) {
		size_t featureId = obj.first;
		double* pXw = (double*)(obj.second);
		(*pSfmFeatures_)[featureId].Xw.x = (float)pXw[0];
		(*pSfmFeatures_)[featureId].Xw.y = (float)pXw[1];
		(*pSfmFeatures_)[featureId].Xw.z = (float)pXw[2];
	}
}

/**
 * @brief Checks the need for and performs re-triangulation on all SFM features.
 *
 * This function identifies all feature IDs from the current SFM feature set and invokes
 * the CheckAndRetrangulation function for these IDs. It is typically called when a
 * global update or a significant adjustment has been made to the camera poses or
 * when new information necessitates a re-evaluation of the 3D points.
 * 
 * @param reprojErrThr The threshold for the reprojection error for triangulation validation.
 * @param cosAngThr The cosine of the threshold angle for triangulation validation.
 */
void Reconstructor::CheckAndRetrangulation(double reprojErrThr, double cosAngThr) {
	// Vector to store the IDs of all SFM features
	std::vector<int> ids;
	for (const auto& sfmfea : *pSfmFeatures_) {
		if (sfmfea.first != -1) {
			ids.push_back(sfmfea.first);
		}
	}

	// Invoke the CheckAndRetrangulation function with the collected feature IDs
	CheckAndRetrangulation(ids, reprojErrThr, cosAngThr);
}

/**
 * @brief Re-triangulates and checks the validity of 3D points for the specified feature IDs.
 *
 * This function performs re-triangulation and validation of 3D points (SFM features) for the given feature IDs.
 * It ensures that the 3D points are accurately reconstructed and meet the defined thresholds for reprojection
 * error and angle. If a feature has not been triangulated yet, it attempts to triangulate it. If a feature has
 * already been triangulated, it checks the validity of the point.
 *
 * @param ids A vector of feature IDs to be re-triangulated and checked.
 * @param reprojErrThr The threshold for the reprojection error for triangulation validation.
 * @param cosAngThr The cosine of the threshold angle for triangulation validation.
 */
void Reconstructor::CheckAndRetrangulation(std::vector<int>& ids, double reprojErrThr, double cosAngThr) {
	// Iterate through each feature ID
	for (auto featureId : ids) {
		// Check if the feature ID exists in the SFM features map
		if (pSfmFeatures_->find(featureId) == pSfmFeatures_->end()) {
			continue;
		}
		auto& sfmfea = (*pSfmFeatures_)[featureId];

		// Attempt to triangulate the feature if it hasn't been triangulated yet
		if (!sfmfea.hasObject) {
			sfmfea.Triangulate(pFrames_, reprojErrThr, cosAngThr);
		}

		// Check the validity of the triangulated point
		if (sfmfea.hasObject) {
			sfmfea.CheckPoint(pFrames_, reprojErrThr, cosAngThr);
		}
	}
}

}