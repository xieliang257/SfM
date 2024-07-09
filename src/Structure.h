#pragma once
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Reconstructor/PoseSolver.h"

namespace sfm {
struct Feature {
	int featureId = -1;
	int imageId = -1;
	cv::Point2f pt;
	cv::Vec3b color;
};

class SFMFeature {
public:
	/**
	 * @brief Triangulates a single 3D point from multiple observations across different frames.
	 *
	 * This method performs robust triangulation using multiple camera views. It extracts the camera
	 * pose and intrinsic matrix for each view where the feature is visible, undistorts the observed
	 * point, and then uses these to triangulate the 3D location of the point. The triangulation is
	 * considered successful if the reprojected error and the cosine of the viewing angle are within
	 * specified thresholds.
	 *
	 * @param pFrames Shared pointer to a vector of Frame objects, each containing image data and metadata.
	 * @param reprojErr Maximum acceptable reprojection error for the triangulation to be considered successful.
	 * @param cosAngleThreshold Maximum acceptable cosine of the viewing angle for the triangulation.
	 * @return True if the triangulation is successful and the results meet the specified criteria, false otherwise.
	 */
	bool Triangulate(const std::shared_ptr<std::vector<Frame>>& pFrames, double reprojErr, double cosAngleThreshold) {
		auto& frames = *pFrames;
		std::vector<cv::Mat> TList, KList;
		std::vector<cv::Point2f> undistortObserves;
		for (const auto& fea : features) {
			int imageId = fea.imageId;
			if (imageId >= frames.size()) {
				continue;
			}

			// If the frame has a valid pose, process the observation.
			if (frames[imageId].hasPose_) {
				TList.push_back(frames[imageId].GetPose().clone());
				KList.push_back(frames[imageId].CameraMat().clone());
				cv::Point2f undistPt;
				frames[imageId].UndistortPoint(fea.pt, undistPt);
				undistortObserves.push_back(undistPt);
			}
		}

		// Require at least two observations for triangulation.
		if (undistortObserves.size() < 2) {
			return false;
		}

		TriangulateResult point;
		// Perform robust triangulation using collected data.
		PoseSolver::RobustTriangulatePoint(KList, TList, undistortObserves, point);

		// Check if the triangulation result meets the specified criteria.
		if (point.statu && point.reprojErr < reprojErr && point.cosAngleView < cosAngleThreshold) {
			Xw = point.object;
			cosViewAngle = point.cosAngleView;
			hasObject = true;
		}
		else {
			return false;
		}
		return true;
	}

	/**
	 * @brief Validates a reconstructed 3D point's accuracy across multiple frames.
	 *
	 * This function checks the reprojection errors and viewing angles of a reconstructed 3D point
	 * against thresholds to determine its validity. It uses camera poses and intrinsic matrices
	 * from each frame where the point was observed. The point is invalidated if it fails these checks.
	 *
	 * @param pFrames Shared pointer to a vector of Frame objects, each containing image data and metadata.
	 * @param reprojErrThreshold Maximum acceptable reprojection error for the point to be considered valid.
	 * @param cosAngleThreshold Maximum acceptable cosine of the viewing angle for the point to be valid.
	 */
	void CheckPoint(const std::shared_ptr<std::vector<Frame>>& pFrames, double reprojErrThreshold, double cosAngleThreshold) {
		auto& frames = *pFrames;
		std::vector<cv::Mat> KList;
		std::vector<cv::Mat> TList;
		std::vector<cv::Point2f> undistObserves;

		// Iterate over each feature observed in the frames.
		for (const auto& fea : features) {
			int imageId = fea.imageId;
			if (imageId >= frames.size()) {
				continue;
			}

			// Collect the observations and the corresponding intrinsic and extrinsic parameters of the camera 
			if (frames[imageId].hasPose_) {
				auto& frame = frames[imageId];
				cv::Point2f undistPt;
				frames[imageId].UndistortPoint(fea.pt, undistPt);
				undistObserves.push_back(undistPt);
				KList.push_back(frames[imageId].K_.clone());
				TList.push_back(frames[imageId].GetPose().clone());
				KList.push_back(frames[imageId].K_.clone());
			}
		}

		// Check and set the view angle if the object has been successfully triangulated.
		if (hasObject) {
			// Counter for the number of inliers.
			int inlierNum = 0;
			for (int k = 0; k < undistObserves.size(); ++k) {
				// Convert 3D point to homogeneous coordinates.
				cv::Mat P = (cv::Mat_<double>(4, 1) << Xw.x, Xw.y, Xw.z, 1);

				// Project the 3D point back to 2D using the camera model.
				cv::Mat Pi = KList[k] * TList[k] * P;

				// Depth value after projection.
				double z = Pi.at<double>(2, 0);
				if (z < 0) {
					// Invalidate the point if it is behind the camera.
					hasObject = false;
				}

				// Normalized x,y coordinate.
				double u = Pi.at<double>(0, 0) / z;
				double v = Pi.at<double>(1, 0) / z;

				// Compute the reprojection error.
				double err = cv::norm(cv::Point2f(u, v) - undistObserves[k]);;
				if (err < reprojErrThreshold) {
					// Increment inlier count if error is within threshold.
					++inlierNum;
				}
			}

			// Invalidate the point if there are not enough inliers.
			if (inlierNum < 2) {
				hasObject = false;
			}

			// Check if the point has a large enough view angle.
			float cosView = PoseSolver::CosViewAngle(TList, Xw);
			if (cosView > cosAngleThreshold) {
				hasObject = false;
			}

			// Update the cosine of the viewing angle.
			cosViewAngle = cosView;
		}
	}


public:
	// An identifier for the feature. Defaulted to -1, indicating that it is uninitialized or invalid.
	int featureId = -1;

	// A boolean flag indicating whether a corresponding 3D object (point) has been successfully triangulated.
	bool hasObject = false;

	// The 3D coordinates of the triangulated point in the world coordinate system.
	cv::Point3f Xw;

	// The cosine of the view angle at which this point is observed. Initialized to 1.
	float cosViewAngle = 1;

	// A collection of observations of this feature across multiple images.
	std::vector<Feature> features;
};
}