#include "Reconstructor/PoseSolver.h"
#include "fstream"

namespace sfm {
void Hat(const cv::Vec3d& q, cv::Mat& Q) {
	double x = q(0);
	double y = q(1);
	double z = q(2);
	if (Q.size() != cv::Size(3, 3)) {
		Q = cv::Mat::zeros(3, 3, CV_64F);
	}
	double qdata[] = { 0, -z, y, z, 0, -x, -y, x, 0 };
	std::memcpy(Q.data, qdata, sizeof(double) * 9);
}

void PoseSolver::TriangulatePoints(const cv::Mat& K1, const cv::Mat& R1, const cv::Mat& t1, 
	                               const cv::Mat& K2, const cv::Mat& R2, const cv::Mat& t2,
	                               const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
	                               std::vector<TriangulateResult>& results) {

	cv::Mat T1(3, 4, R1.type());
	cv::Mat T2(3, 4, R2.type());
	R1.copyTo(T1(cv::Rect(0, 0, 3, 3)));
	t1.copyTo(T1(cv::Rect(3, 0, 1, 3)));
	R2.copyTo(T2(cv::Rect(0, 0, 3, 3)));
	t2.copyTo(T2(cv::Rect(3, 0, 1, 3)));
	std::vector<cv::Mat> KList = { K1, K2 };
	std::vector<cv::Mat> TList = { T1, T2 };
	results.clear();
	for (int i = 0; i < pts1.size(); ++i) {
		std::vector<cv::Point2f> pts = { pts1[i],pts2[i] };
		TriangulateResult result;
		TriangulatePoint(KList, TList, pts, result);
		results.push_back(result);
	}
}

/**
 * Calculates the minimum cosine of the viewing angles between multiple camera views for a given 3D point.
 *
 * @param projMatList A vector of projection matrices representing the cameras. 
 *                    Each matrix transforms 3D world coordinates to camera coordinates.
 * @param object A cv::Point3f object representing the 3D point whose view angles are to be calculated.
 * @return Returns the smallest cosine of the viewing angles between any two camera views of the given point.
 */
float PoseSolver::CosViewAngle(const std::vector<cv::Mat>& projMatList, const cv::Point3f& object) {
	double minCosView = 1;
	// Convert the 3D point into a Mat object.
	cv::Mat Pw = (cv::Mat_<double>(3, 1) << object.x, object.y, object.z);

	// This will hold the normalized direction vectors from each camera to the point.
	std::vector<cv::Mat> views;
	for (int i = 0; i < projMatList.size(); ++i) {
		// Calculate the direction vector from camera to the point.
		cv::Mat view = Pw - (-projMatList[i](cv::Rect(0, 0, 3, 3)).t() * projMatList[i](cv::Rect(3, 0, 1, 3)));
		cv::normalize(view, view);
		views.push_back(view.clone());
	}

	// Compute the minimum cosine of the angle between each pair of views.
	for (int i = 1; i < views.size(); ++i) {
		for (int j = 0; j < i; ++j) {
			double cosVal = cv::norm(views[i].t() * views[j]);
			minCosView = std::min(cosVal, minCosView);
		}
		
	}
	return minCosView;
}

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
void PoseSolver::TriangulatePoint(const std::vector<cv::Mat>& KList, const std::vector<cv::Mat>& TList,
	                              const std::vector<cv::Point2f>& pts, TriangulateResult& result) {
	if (pts.size() < 2) {
		result.statu = 0;
		return;
	}

	// Triangulates a 3D point from multiple camera views by minimizing the sum of the squared dot products between
	// the unit vectors of observed points and the vectors from the cameras to the 3D point.
	cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);
	for (int i = 0; i < TList.size(); ++i) {
		cv::Mat Q;
		cv::Vec3d q(pts[i].x, pts[i].y, 1);
		cv::normalize(q, q);
		Hat(q, Q);
		cv::Mat C = Q *KList[i] * TList[i];
		A += C.t() * C;
	}
	cv::Mat W, U, VT;
	cv::SVDecomp(A, W, U, VT, cv::SVD::FULL_UV);
	double x = VT.at<double>(3, 0);
	double y = VT.at<double>(3, 1);
	double z = VT.at<double>(3, 2);
	double tmp = VT.at<double>(3, 3);
	if (fabs(tmp) < 1e-8) {
		tmp = (tmp < 0) ? -1e-8 : 1e-8;
	}
	x /= tmp;
	y /= tmp;
	z /= tmp;
	cv::Point3f object = cv::Point3f(x, y, z);
	result.object = object;

	// Compute the average reprojection error.
	double reprojErr = 0;
	for (int i = 0; i < TList.size(); ++i) {
		cv::Mat P = (cv::Mat_<double>(4, 1) << x, y, z, 1);
		cv::Mat q = KList[i] * TList[i] * P;
		double tmp = q.at<double>(2, 0);
		if (fabs(tmp) < 1e-8) {
			tmp = (tmp < 0) ? -1e-8 : 1e-8;
		}
		double u = q.at<double>(0, 0) / tmp;
		double v = q.at<double>(1, 0) / tmp;
		reprojErr += cv::norm(cv::Point2f(u, v) - pts[i]);
	}
	reprojErr /= pts.size();
	result.reprojErr = reprojErr;

	// Check if the point is in front of all cameras.
	for (int k = 0; k < pts.size(); ++k) {
		cv::Mat P = (cv::Mat_<double>(4, 1) << object.x, object.y, object.z, 1);
		cv::Mat Pcurr = KList[k]*TList[k] * P;
		double z = Pcurr.at<double>(2, 0);
		if (z < 0) {
			result.statu = 0;
			return;
		}
	}

	// Compute the cosine of the smallest viewing angle.
	double cosView = CosViewAngle(TList, object);
	result.cosAngleView = cosView;
	result.statu = 1;
}

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
void PoseSolver::RobustTriangulatePoint(const std::vector<cv::Mat>& KList, const std::vector<cv::Mat>& TList,
	                                    const std::vector<cv::Point2f>& pts, TriangulateResult& result) {
	// Triangulate directly if only two views are available.
	if (KList.size() == 2) {
		TriangulatePoint(KList, TList, pts, result);
		return;
	}

	result.statu = false;
	double minAvgErr = 1e8;
	for (int i = 0; i < TList.size() - 1; ++i) {
		// Select 2 neighbor cameras as a pair
		std::vector<cv::Mat> subTList = { TList[i],TList[i + 1] };
		std::vector<cv::Mat> subKList = { KList[i],KList[i + 1] };
		std::vector<cv::Point2f> subPts = { pts[i],pts[i + 1] };
		TriangulateResult subResult;
		TriangulatePoint(subKList, subTList, subPts, subResult);
		if (!subResult.statu) {
			continue;
		}

		// Project the triangulated point into all views and calculate the error
		cv::Mat P = (cv::Mat_<double>(4, 1) << subResult.object.x, subResult.object.y, subResult.object.z, 1);
		double threshold = 3, sumErr = 0, sumWeight = 0;
		for (int j = 0; j < TList.size(); ++j) {
			cv::Mat Pi = KList[j]* TList[j] * P;
			double u = Pi.at<double>(0, 0) / Pi.at<double>(2, 0);
			double v = Pi.at<double>(1, 0) / Pi.at<double>(2, 0);
			double err = cv::norm(cv::Point2f(u, v) - pts[j]);
			double w = err / threshold;
			w = std::min(w, 1.);
			sumErr += w * err;
			sumWeight += w;
		}
		double avgErr = sumErr / (sumWeight + 1e-8);
		if (minAvgErr > avgErr) {
			minAvgErr = avgErr;
			result = subResult;
		}
	}
}

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
int PoseSolver::EssentialSolver(const cv::Mat& K,
								const std::vector<cv::Point2f>& pts1,
								const std::vector<cv::Point2f>& pts2,
								cv::Mat& R_2_1, cv::Mat& t_2_1, std::vector<TriangulateResult>& points,
	                            double threshold) {
	std::vector<uchar> inliers;
	cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, threshold, inliers);
	cv::Mat R1, R2, t;
	cv::decomposeEssentialMat(E, R1, R2, t);
	cv::Mat _t = -t;
	cv::Mat R0 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);
	std::vector<TriangulateResult> results1, results2, results3, results4;
	TriangulatePoints(K, R0, t0, K, R1, t, pts1, pts2, results1);
	TriangulatePoints(K, R0, t0, K, R2, t, pts1, pts2, results2);
	TriangulatePoints(K, R0, t0, K, R1, _t, pts1, pts2, results3);
	TriangulatePoints(K, R0, t0, K, R2, _t, pts1, pts2, results4);
	int frontCnt1 = 0, frontCnt2 = 0, frontCnt3, frontCnt4;
	double avgErr1 = 0, avgErr2 = 0, avgErr3 = 0, avgErr4 = 0;
	double w1 = 0, w2 = 0;

	// Lambda function to check and evaluate each camera pose(R|t)
	auto CheckTriangulator = [](std::vector<TriangulateResult>& points, cv::Mat& R, cv::Mat& t, int& frontCnt, double& avgErr) {
		frontCnt = 0;
		double w1 = 0;
		for (int i = 0; i < points.size(); ++i) {
			if (!points[i].statu) {
				continue;
			}
			double z1 = points[i].object.z;
			cv::Mat P1 = (cv::Mat_<double>(3, 1) << points[i].object.x, points[i].object.y, points[i].object.z);
			cv::Mat P2 = R * P1 + t;
			double z2 = P2.at<double>(2, 0);
			if (z1 > 0 && z2 > 0) {
				frontCnt++;
				double w = 1;
				if (points[i].reprojErr > 10) {
					w = 1 / (points[i].reprojErr - 9);
				}
				w = std::min(1., w);
				w1 += w;
				avgErr += points[i].reprojErr * w;
			}
		}
		avgErr /= w1;
	};
	
	// Evaluate each pose(R|t)
	CheckTriangulator(results1, R1, t, frontCnt1, avgErr1);
	CheckTriangulator(results2, R2, t, frontCnt2, avgErr2);
	CheckTriangulator(results3, R1, _t, frontCnt3, avgErr3);
	CheckTriangulator(results4, R2, _t, frontCnt4, avgErr4);

	// Find the pose(R|t) with the highest count of points in front
	int maxCnt = 0;
	maxCnt = std::max(std::max(std::max(frontCnt1, frontCnt2), frontCnt3), frontCnt4);
	if (maxCnt == frontCnt1) {
		R1.copyTo(R_2_1);
		t.copyTo(t_2_1);
		points = results1;
	}
	else if (maxCnt == frontCnt2) {
		R2.copyTo(R_2_1);
		t.copyTo(t_2_1);
		points = results2;
	}
	else if (maxCnt == frontCnt3) {
		R1.copyTo(R_2_1);
		_t.copyTo(t_2_1);
		points = results3;
	}
	else {
		R2.copyTo(R_2_1);
		_t.copyTo(t_2_1);
		points = results4;
	}
	return maxCnt;
}

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
bool PoseSolver::EssentialSolver(const cv::Mat& K,
	                             const std::vector<double>& depths1,
	                             const std::vector<cv::Point2f>& pts1,
	                             const std::vector<cv::Point2f>& pts2,
	                             cv::Mat& R_2_1, cv::Mat& t_2_1, std::vector<TriangulateResult>& points,
	                             double threshold) {
	// Estimate the camera pose from corresponding points using essential matrix decomposition and perform triangulation.
	EssentialSolver(K, pts1, pts2, R_2_1, t_2_1, points, threshold);
	TriangulatePoints(K, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, R_2_1, t_2_1, pts1, pts2, points);
	std::vector<double> scaleList;
	int validCnt = 0;

	// Iterate over triangulated points to compute scale factors.
	for (int i = 0; i < depths1.size(); ++i) {
		if (points[i].object.z && points[i].statu != 0) {
			++validCnt;
		}
		if (depths1[i] < 1e-8) {
			continue;
		}
		if (!points[i].statu || points[i].object.z < 1e-8) {
			continue;
		}
		double d = points[i].object.z;
		double s = depths1[i] / d;
		scaleList.push_back(s);
	}

	if (scaleList.size() == 0) {
		return false;
	}

	// Check if there are enough valid points.
	if (validCnt < pts1.size() * 0.7) {
		return false;
	}
	std::sort(scaleList.begin(), scaleList.end());
	// Median scale.
	double scale = scaleList[scaleList.size() / 2];

	// Ensure that the scale factors are not too divergent.
	if (scaleList[scaleList.size() * 0.55] > scaleList[scaleList.size() * 0.45] * 1.5) {
		return false;
	}

	// Apply the determined scale to the translation vector and 3D points.
	t_2_1 *= scale;
	for (auto& p : points) {
		p.object *= scale;
	}

	return true;
}

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
bool PoseSolver::HomographySolver(const cv::Mat& K,
	                              const std::vector<double>& depths1,
	                              const std::vector<cv::Point2f>& pts1,
	                              const std::vector<cv::Point2f>& pts2,
	                              cv::Mat& R_2_1, cv::Mat& t_2_1,
	                              double threshold) {
	double fx = K.at<double>(0, 0);
	double fy = K.at<double>(1, 1);
	double cx = K.at<double>(0, 2);
	double cy = K.at<double>(1, 2);
	std::vector<uchar> inliers;
	// Compute homography matrix using RANSAC
	cv::Mat H = cv::findHomography(pts1, pts2, inliers, cv::RANSAC, threshold);

	// Decompose homography matrix to obtain possible rotations, translations, and normals
	std::vector<cv::Mat> RList, translations, normals;
	cv::decomposeHomographyMat(H, K, RList, translations, normals);

	std::vector<std::vector<double>> errsList(RList.size());
	std::vector<int> cntList;
	int maxCnt = 0;

	for (int i = 0; i < RList.size(); ++i) {
		std::vector<TriangulateResult> points;
		// Perform triangulation using each rotation and translation
		TriangulatePoints(K, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, RList[i], translations[i], pts1, pts2, points);
		int cnt = 0;
		for (auto p : points) {
			if (p.object.z > 0) {
				++cnt;
			}
		}
		cntList.push_back(cnt);
		maxCnt = std::max(maxCnt, cnt);

		double epsilon = 1e-6;
		// Calculate scale factors for translation adjustment
		double scale = 1;
		std::vector<double> scaleList;
		for (int j = 0; j < depths1.size(); ++j) {
			if (depths1[j] > epsilon && points[j].statu && points[j].object.z > epsilon) {
				double s = depths1[j] / points[j].object.z;
				scaleList.push_back(s);
			}
		}
		if (scaleList.size() > 0) {
			std::sort(scaleList.begin(), scaleList.end());
			if (scaleList[scaleList.size() * 0.55] > scaleList[scaleList.size() * 0.45] * 1.5) {
				return false;
			}
			scale = scaleList[scaleList.size() / 2];
		}
		translations[i] *= scale;

		// Calculate reprojection errors
		for (int j = 0; j < pts1.size(); ++j) {
			if (depths1[j] > epsilon) {
				cv::Mat P = (cv::Mat_<double>(3, 1) << (pts1[j].x-cx)/fx, (pts1[j].y-cy)/fy, 1);
				P *= depths1[j];
				cv::Mat P2 = K*(RList[i] * P + translations[i]);
				double x = P2.at<double>(0, 0) / P2.at<double>(2, 0);
				double y = P2.at<double>(1, 0) / P2.at<double>(2, 0);
				double e = cv::norm(pts2[j] - cv::Point2f(x, y));
				errsList[i].push_back(e);
			}
		}
	}

	double minErr = 1e8;
	int bestId = -1;
	// Find the best decomposition based on median reprojection error
	for (int i = 0; i < RList.size(); ++i) {
		if (errsList[i].size() == 0) {
			continue;
		}
		std::sort(errsList[i].begin(), errsList[i].end());
		int midId = errsList.size() *0.8;
		cv::Vec3d r;
		cv::Rodrigues(RList[i], r);
		if (errsList[i][midId] < minErr && cntList[i]>=maxCnt*0.8) {
			minErr = errsList[i][midId];
			bestId = i;
		}
	}
	if (bestId == -1) {
		return false;
	}
	
	// Check if the number of valid points is sufficient
	if (cntList[bestId] < pts1.size() * 0.5) {
		return false;
	}

	R_2_1 = RList[bestId].clone();
	t_2_1 = translations[bestId].clone();
	return true;
}

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
bool PoseSolver::PnPSolver(const std::vector<cv::Point3f>& _objects,
	                       const std::vector<cv::Point2f>& _observes,
	                       const cv::Mat& K,
	                       cv::Mat& R, cv::Mat& t) {
	// Lambda function to remove outliers based on the distance between points
	auto RemoveOutliers = [](std::vector<cv::Point3f>& objects, std::vector<cv::Point2f>& observes) {
		std::vector<double> errors;
		for (int i = 0; i < objects.size(); ++i) {
			std::vector<double> localErrs;
			for (int j = 0; j < objects.size(); ++j) {
				if (i == j) {
					continue;
				}
				double e = cv::norm(objects[i] - objects[j]);
				localErrs.push_back(e);
			}
			if (localErrs.size() > 3) {
				std::sort(localErrs.begin(), localErrs.end());
				// Use the 4th smallest distance as a representative error
				errors.push_back(localErrs[3]);
			}
			else {
				// Assign a large error if there are not enough points
				errors.push_back(1e6);
			}
		}

		// Sort errors to determine a threshold for outlier removal
		auto sortErrs = errors;
		std::sort(sortErrs.begin(), sortErrs.end());
		if (sortErrs.size() > 10) {
			// Use the 90th percentile as the threshold
			double threshold = sortErrs[sortErrs.size() * 0.9];
			int id = 0;
			for (int i = 0; i < objects.size(); ++i) {
				if (errors[i] <= threshold) {
					objects[id] = objects[i];
					observes[id] = observes[i];
					++id;
				}
			}
			// Resize vectors to remove outliers
			objects.resize(id);
			observes.resize(id);
		}
	};

	// Remove outliers
	auto objects = _objects;
	auto observes = _observes;
	RemoveOutliers(objects, observes);

	// Solve PnP using RANSAC to estimate the initial pose
	cv::Mat rvec_i_0, tvec_i_0;
	std::vector<int> pnpinliers;
	cv::solvePnPRansac(objects, observes, K, cv::Mat(), rvec_i_0, tvec_i_0, false, 1000, 8., 0.999, pnpinliers, cv::SOLVEPNP_EPNP);

	// Collect inlier points identified by RANSAC
	std::vector<cv::Point3f> inlierObjects;
	std::vector<cv::Point2f> inlierObserves;
	for (auto id : pnpinliers) {
		inlierObjects.push_back(objects[id]);
		inlierObserves.push_back(observes[id]);
	}

	// If not enough inliers, return false
	if (inlierObjects.size() < 30) {
		return false;
	}

	// Refine the pose estimation using inlier points and Levenberg-Marquardt optimization
	cv::solvePnPRefineLM(inlierObjects, inlierObserves, K, cv::Mat(), rvec_i_0, tvec_i_0);

	// Set output rotation matrix and translation vector
	cv::Rodrigues(rvec_i_0, R);
	tvec_i_0.copyTo(t);
	return true;
}

}