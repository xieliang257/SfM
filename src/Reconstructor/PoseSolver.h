#pragma once
#include <opencv2/opencv.hpp>

namespace sfm {
struct TriangulateResult {
	cv::Point3f object;
	uchar statu = 0;
	double reprojErr = 1e6;
	double cosAngleView = 1;
};

/**
 * PoseSolver is a utility class that provides various methods for estimating the pose of a camera relative to 3D points.
 * The class includes functions for solving the essential matrix, homography matrix, and Perspective-n-Point (PnP) problem,
 * as well as methods for robustly triangulating 3D points from multiple views.
 */
class PoseSolver {
public:
	/**
	 * Calculates the minimum cosine of the viewing angles between multiple camera views for a given 3D point.
	 *
	 * @param projMatList A vector of projection matrices representing the cameras.
	 *                    Each matrix transforms 3D world coordinates to camera coordinates.
	 * @param object A cv::Point3f object representing the 3D point whose view angles are to be calculated.
	 * @return Returns the smallest cosine of the viewing angles between any two camera views of the given point.
	 */
	static float CosViewAngle(const std::vector<cv::Mat>& projMatList, const cv::Point3f& object);

	static void TriangulatePoints(const cv::Mat& K1, const cv::Mat& R1, const cv::Mat& t1,
		                          const cv::Mat& K2, const cv::Mat& R2, const cv::Mat& t2,
		                          const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
		                          std::vector<TriangulateResult>& results);

	/**
	 * Performs robust triangulation of a 3D point from its 2D projections across multiple camera views.
	 * This method attempts to find the best triangulation result by iteratively using pairs of cameras and
	 * selecting the result with the minimum average reprojection error.
	 *
	 * @param KList A vector of camera intrinsic matrices for all views.
	 * @param TList A vector of camera extrinsic matrices (transformation from world to camera coordinates).
	 * @param pts A vector of 2D points corresponding to the projections of the 3D point in each camera.
	 * @param result A structure to store the output including the triangulated result.
	 */
	static void RobustTriangulatePoint(const std::vector<cv::Mat>& KList, const std::vector<cv::Mat>& TList,
		                               const std::vector<cv::Point2f>& pts, TriangulateResult& result);

	/**
	 * Triangulates a single 3D point from its projections in multiple camera views using the linear method.
	 *
	 * @param KList Vector of intrinsic camera matrices (3x3 matrix for each camera).
	 * @param TList Vector of transformation matrices (3x4 matrix for each camera that converts from world to camera coordinates).
	 * @param pts Vector of image points (2D points in image coordinates for each camera view).
	 * @param result Reference to a TriangulateResult struct that will hold the output.
	 *
	 * The function ensures the triangulated point is in front of every camera and evaluates the reprojection error and the viewing angle to ensure high-quality triangulation.
	 */
	static void TriangulatePoint(const std::vector<cv::Mat>& KList, const std::vector<cv::Mat>& TList,
		                         const std::vector<cv::Point2f>& pts, TriangulateResult& result);

	/**
	 * Solves the essential matrix and performs robust triangulation using four possible camera pose solutions.
	 *
	 * This function computes the essential matrix using matched points from two views and then decomposes it to obtain possible
	 * rotation (R) and translation (t) solutions. It triangulates points using each solution and evaluates them based on the number
	 * of points that triangulate in front of both cameras and the reprojection error.
	 *
	 * @param K The camera intrinsic matrix used for computing the essential matrix.
	 * @param pts1 The vector of 2D points in the first image.
	 * @param pts2 The vector of 2D points in the second image.
	 * @param R_2_1 The output rotation matrix that describes the rotation from the first to the second camera.
	 * @param t_2_1 The output translation vector that describes the translation from the first to the second camera.
	 * @param points The vector of triangulated points that resulted in the best solution.
	 * @param threshold The RANSAC threshold used when computing the essential matrix.
	 * @return Returns the maximum number of points that are in front of both cameras for the best solution.
	 */
	static int EssentialSolver(const cv::Mat& K,
		                       const std::vector<cv::Point2f>& pts1,
		                       const std::vector<cv::Point2f>& pts2,
		                       cv::Mat& R_2_1, cv::Mat& t_2_1, std::vector<TriangulateResult>& points,
		                       double threshold);

	/**
	 * Refines the pose estimation between two views using the essential matrix and then adjusts the scale of the triangulated points
	 * based on depth information from the first view.
	 *
	 * @param K The intrinsic camera matrix used for both views.
	 * @param depths1 A vector containing the depth information for each point in the first view.
	 * @param pts1 A vector of 2D points from the first view.
	 * @param pts2 A vector of 2D points from the second view.
	 * @param R_2_1 The rotation matrix output that aligns the first view to the second view.
	 * @param t_2_1 The translation vector output that aligns the first view to the second view.
	 * @param points A vector of triangulation results that will store the 3D positions of points as seen from the first view.
	 * @param threshold The RANSAC threshold used when computing the essential matrix.
	 *
	 * Returns:
	 * - Returns true if a valid scale is found and applied, false otherwise.
	 */
	static bool EssentialSolver(const cv::Mat& K,
		                        const std::vector<double>& depths1,
		                        const std::vector<cv::Point2f>& pts1,
		                        const std::vector<cv::Point2f>& pts2,
		                        cv::Mat& R_2_1, cv::Mat& t_2_1, std::vector<TriangulateResult>& points,
		                        double threshold);

	/**
	 * Estimates the camera pose between two views using homography decomposition and then performs triangulation to select the best one.
	 *
	 * @param K The intrinsic camera matrix.
	 * @param depths1 A vector containing the depth information for each point in the first view.
	 * @param pts1 A vector of 2D points from the first view.
	 * @param pts2 A vector of 2D points from the second view.
	 * @param R_2_1 The output rotation matrix that aligns the first view with the second view.
	 * @param t_2_1 The output translation vector that aligns the first view with the second view.
	 * @param threshold The RANSAC threshold used when computing the homography matrix.
	 *
	 * Returns:
	 * - Returns true if a valid pose estimation is found, false otherwise.
	 */
	static bool HomographySolver(const cv::Mat& K,
		                         const std::vector<double>& depths1,
		                         const std::vector<cv::Point2f>& pts1,
		                         const std::vector<cv::Point2f>& pts2,
		                         cv::Mat& R_2_1, cv::Mat& t_2_1,
		                         double threshold);

	/**
	 * Estimates the camera pose using the Perspective-n-Point (PnP) algorithm and removes outliers to improve accuracy.
	 *
	 * @param _objects A vector of 3D points in the world coordinate system.
	 * @param _observes A vector of corresponding 2D points on the image.
	 * @param K The camera intrinsic matrix.
	 * @param R The output rotation matrix that from the world coordinate system to the camera coordinate system.
	 * @param t The output translation vector that from the world coordinate system to the camera coordinate system.
	 *
	 * Returns:
	 * - Returns true if the pose estimation is successful and there are enough inlier points for refinement, false otherwise.
	 */
	static bool PnPSolver(const std::vector<cv::Point3f>& objects,
		                  const std::vector<cv::Point2f>& observes,
		                  const cv::Mat& K,
		                  cv::Mat& R, cv::Mat& t);
	
};
}