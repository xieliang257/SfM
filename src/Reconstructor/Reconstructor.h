#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ImageProcessor/ImageProcessor.h"
#include "FeatureManager/FeatureManager.h"

namespace sfm {
/**
* @brief Class for performing structure-from-motion (SfM) reconstruction.
*
* This class manages the entire SfM reconstruction process, from initial setup to final optimization,
* including loading configurations, handling input frames and matches, and running bundle adjustments.
*/
class Reconstructor {
public:
	/**
	 * @brief Constructor for the Reconstructor class.
	 * @param configFile Path to the configuration file with intrinsic camera parameters and other settings.
	 * @param workDir Working directory for storing intermediate results.
	 */
	Reconstructor(const std::string& configFile, const std::string& workDir);

	/**
	 * @brief Reconstruct the structure from motion (SfM) using input frames and matches.
	 *
	 * This function initializes the reconstruction process, iterates through each frame incrementally,
	 * and performs global bundle adjustment periodically to optimize the camera poses and structure.
	 *
	 * @param pFrames A shared pointer to a vector of Frame objects, which represent the input images and their properties.
	 * @param pMatches A shared pointer to AllMatchesType, which encapsulates all the feature matches between frames.
	 */
	void Reconstruct(const std::shared_ptr<std::vector<Frame>>& pFrames,
		     const std::shared_ptr<AllMatchesType>& pMatches);
	
	std::shared_ptr<std::map<size_t, SFMFeature>>& SfmFeaturesPtr() {
		return pSfmFeatures_;
	}

private:

	/**
	 * @brief Initializes the structure from motion (SfM) reconstruction process by finding and setting the initial pair of frames.
	 *
	 * This function attempts to find the first pair of frames that will serve as the base for the entire reconstruction process.
	 * If the first pair is found successfully, it sets their initial poses and attempts to triangulate initial features.
	 *
	 * @return Returns true if the initial frame pair is successfully found and initialized; otherwise, returns false.
	 */
	bool Start();

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
	bool FindFirstPair(int& imageId1, int& imageId2, cv::Mat& R, cv::Mat& t);

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
	int Increment();

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
	bool PnPPose(int frameId, cv::Mat& R_i_w, cv::Mat& t_i_w);

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
	bool EHPose(int frameId, cv::Mat& R_i_w, cv::Mat& t_i_w);

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
	void LocalBA(int incrementId);

	/**
	 * @brief Performs global bundle adjustment across all frames in the reconstruction.
	 *
	 * Unlike LocalBA, GlobalBA optimizes the entire structure and camera parameters throughout
	 * the entire dataset to ensure overall consistency and accuracy. It includes the first and
	 * second frames as constants to anchor the global solution and minimize drift.
	 */
	void GlobalBA();

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
	void BAImplement(const std::vector<int>& constantFrameIds);

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
	void CheckAndRetrangulation(double reprojErrThr, double cosAngThr);

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
	void CheckAndRetrangulation(std::vector<int>& ids, double reprojErrThr, double cosAngThr);

private:
	// Working directory for output and intermediate files.
	std::string workDir_;

	// Manages feature extraction and matching.
	std::shared_ptr<FeatureManager> featureManager_ = nullptr;

	// Pointer to a vector of frames used in SfM.
	std::shared_ptr<std::vector<Frame>> pFrames_ = nullptr;

	// Pointer to all pairwise feature matches
	std::shared_ptr<AllMatchesType> pMatches_ = nullptr;

	// Pointer to all SFM features mapped by IDs.
	std::shared_ptr<std::map<size_t, SFMFeature>> pSfmFeatures_ = nullptr;

	// IDs of features used for local bundle adjustment.
	std::vector<int> localFeaturesId_;

	// IDs of the first and second frame in the initial pair.
	std::pair<int, int> startPair_;

	// Number of intrinsic parameters used, 3(f, cx, cy) by default.
	size_t intrinsicsSize_ = 3;

	// Flag to optimize intrinsic parameters.
	bool optimizeIntrinsicParams_ = true;

	// Flag to use shared intrinsics across all frames.
	bool shareIntrinsicParams_ = true;
	
	// List of frames sorted by their potential contribution to the reconstruction.
	std::vector<std::pair<int,int>> viewsList_;
};
}