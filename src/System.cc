#include <string>
#include <fstream>
#include <iomanip>
#include "System.h"

namespace sfm {
System::System(const std::string& configFile, const std::string& workDir) {
	workDir_ = workDir;
	imageProcessor_ = std::make_shared<ImageProcessor>(configFile, workDir_);
	reconstructor_ = std::make_shared<Reconstructor>(configFile, workDir_);
}

/**
 * @brief Writes 3D points and camera representations to a PLY file.
 *
 * This function exports a combined set of 3D points and camera frustums to a PLY file,
 * which is commonly used for storing 3D data. Each point and camera position is written with
 * associated color data, and connections between camera positions are represented as edges.
 *
 * @param filename Path to the output PLY file.
 * @param pts Vector of 3D points (cv::Vec3d), each representing a vertex in the 3D space.
 * @param colors Vector of RGB colors (cv::Vec3b) corresponding to each 3D point.
 * @param cams Vector of vectors, where each inner vector represents the vertices of a camera frustum in 3D space.
 */
void WritePly(const std::string& filename,
              const std::vector<cv::Vec3d>& pts,
              const std::vector<cv::Vec3b>& colors,
              const std::vector<std::vector<cv::Point3f>>& cams) {
              std::ofstream plyFile;
	plyFile.open(filename);

	// Writing the header of the PLY file, specifying format, elements, and properties
	plyFile << "ply\n";
	plyFile << "format ascii 1.0\n";
	plyFile << "element vertex " << pts.size() + cams.size() * 5 << "\n"; // Total vertices from points and cameras
	plyFile << "property float x\n";
	plyFile << "property float y\n";
	plyFile << "property float z\n";
	plyFile << "property uchar red\n";
	plyFile << "property uchar green\n";
	plyFile << "property uchar blue\n";
	plyFile << "element edge " << cams.size() * 8 << "\n"; // Edges formed by camera frustums
	plyFile << "property int vertex1\n";
	plyFile << "property int vertex2\n";
	plyFile << "end_header\n";

	// Writing vertex data: 3D points followed by their colors
	for (size_t i = 0; i < pts.size(); ++i) {
		plyFile << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " ";
		plyFile << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << "\n";
	}

	// Writing camera vertices with fixed red color to distinguish them
	// Offset for camera vertex indices
	int indexOffset = pts.size();
	for (const auto& cam : cams) {
		for (const auto& point : cam) {
			plyFile << point.x << " " << point.y << " " << point.z << " ";
			// Red color for camera vertices
			plyFile << "255 0 0\n"; // 
		}
	}

	// Writing edge data for camera frustums to represent their geometric shape
	for (int i = 0; i < cams.size(); ++i) {
		// Starting index for each camera's vertices
		int baseIndex = indexOffset + i * 5;

		// Connect each camera vertex to form the frustum shape
		plyFile << baseIndex + 0 << " " << baseIndex + 1 << "\n";
		plyFile << baseIndex + 0 << " " << baseIndex + 2 << "\n";
		plyFile << baseIndex + 0 << " " << baseIndex + 3 << "\n";
		plyFile << baseIndex + 0 << " " << baseIndex + 4 << "\n";
		plyFile << baseIndex + 1 << " " << baseIndex + 2 << "\n";
		plyFile << baseIndex + 2 << " " << baseIndex + 3 << "\n";
		plyFile << baseIndex + 3 << " " << baseIndex + 4 << "\n";
		plyFile << baseIndex + 4 << " " << baseIndex + 1 << "\n";
	}

	plyFile.close();
}

/**
 * Computes the distance from each point in a set to its k-th nearest neighbor using the FLANN library.
 *
 * @param points A vector of cv::Vec3d points representing the dataset in which each point's k-th nearest neighbor distance will be calculated.
 * @param k The number of nearest neighbors to consider, the function returns the distance to the k-th nearest one.
 * @param distances Output vector where each element corresponds to the k-th nearest neighbor distance of the corresponding point in the input vector.
 */
void DistanceKNN(const std::vector<cv::Vec3d>& points, int k, std::vector<double>& distances) {
	int nPoints = points.size();
	distances.clear();
	if (nPoints == 0) {
		return;
	}
	k = std::min(k, nPoints);
	cv::Mat dataset(nPoints, 3, CV_32F);
	for (int i = 0; i < points.size(); ++i) {
		dataset.at<float>(i, 0) = points[i](0);
		dataset.at<float>(i, 1) = points[i](1);
		dataset.at<float>(i, 2) = points[i](2);
	}

	// Create a kd-tree index using the dataset.
	cv::flann::Index kdtree(dataset, cv::flann::KDTreeIndexParams(1));

	distances.clear();
	distances.reserve(nPoints);
	std::vector<int> indices(k);
	std::vector<float> dists(k);

	// Perform k-NN search and store the distance to the k-th nearest neighbor.
	for (int i = 0; i < nPoints; ++i) {
		kdtree.knnSearch(dataset.row(i), indices, dists, k, cv::flann::SearchParams(64));
		distances.push_back(dists.back());
	}
}

/**
 * @brief Exports the 3D points and camera positions to a PLY file.
 *
 * This function gathers 3D points from SFM features and their associated color information,
 * along with the camera frustum positions. It filters points based on visibility and the number
 * of features. The results are then written to a PLY file for visualization or further processing.
 *
 * @param dirPath The directory path where the output PLY file will be saved.
 * @param pSfmFeatures Shared pointer to a map containing SFM features. Each feature includes
 *                     3D coordinates, color, and associated image features.
 */
void System::OutputPointCloud(const std::string& dirPath, const std::shared_ptr<std::map<size_t, SFMFeature>>& pSfmFeatures) {
	const auto& sfmFeatures = *pSfmFeatures;
	const auto& frames = *imageProcessor_->FramePtr();
	std::vector<cv::Vec3d> objects;
	std::vector<cv::Vec3b> colors;

	// Threshold for filtering points based on the viewing angle
	double threshold = cos(1 * CV_PI / 180);

	// Iterate through each feature and select those that meet the visibility and quantity criteria
	for (const auto& fea : sfmFeatures) {
		if (fea.second.hasObject && fea.second.cosViewAngle < threshold && fea.second.features.size()>=2) {
			cv::Point3f p = fea.second.Xw;
			cv::Point2f q = fea.second.features.front().pt;
			cv::Vec3b c = fea.second.features.front().color;
			objects.push_back(cv::Vec3d(p.x, p.y, p.z));
			colors.push_back(c);
		}
	}

	// Outlier removal based on the distance to the knn in a set of 3D points.
	// Skip when there are too few points for a meaningful KNN estimate.
	if (objects.size() > 5) {
		std::vector<double> disList;
		DistanceKNN(objects, 5, disList);
		auto sortList = disList;
		std::sort(sortList.begin(), sortList.end());
		const size_t thrIdx = std::min(static_cast<size_t>(sortList.size() * 0.98), sortList.size() - 1);
		double thr = sortList[thrIdx];
		int validId = 0;
		for (int i = 0; i < disList.size(); ++i) {
			if (disList[i] <= thr) {
				objects[validId] = objects[i];
				colors[validId] = colors[i];
				++validId;
			}
		}
		objects.resize(validId);
		colors.resize(validId);
	}

	std::vector<std::vector<cv::Point3f>> cams;
	// Iterate through frames and calculate camera frustum corners based on pose
	//for (const auto& frm : frames) {
	//	if (frm.hasPose_) {
	//		std::vector<cv::Point3f> camPts = { cv::Point3f(0,0,0),cv::Point3f(-1,-1, 1), cv::Point3f(1,-1,1), cv::Point3f(1,1,1),cv::Point3f(-1,1,1) };
	//		for (auto& p : camPts) {
	//			p *= 0.2;
	//			cv::Mat mp(3, 1, CV_64F);
	//			mp.at<double>(0, 0) = p.x;
	//			mp.at<double>(1, 0) = p.y;
	//			mp.at<double>(2, 0) = p.z;
	//			cv::Mat pw = frm.R_i_w_.t() * (mp - frm.t_i_w_);
	//			p.x = pw.at<double>(0, 0);
	//			p.y = pw.at<double>(1, 0);
	//			p.z = pw.at<double>(2, 0);
	//		}
	//		cams.push_back(camPts);
	//	}
	//}

	const auto fileName = dirPath + "/sfmpts.ply";
	WritePly(fileName, objects, colors, cams);
	std::cout << "point cloud was writed at: " << fileName << std::endl;
}

/**
 * @brief Executes the complete Structure from Motion (SfM) process.
 *
 * This method drives the entire SfM pipeline. It starts by processing images to extract and match features.
 * It then reconstructs the 3D scene using these features and finally outputs the resulting point cloud
 * and camera positions to a PLY file. Timing for each major step is recorded and output to give insights
 * into the performance of each stage.
 *
 * @param imageDir Directory containing the images to be processed.
 */
void System::RunSFM(const std::string& imageDir) {
	auto t0 = cv::getTickCount();

	// Start the viewer up front so feature extraction and matching can be visualized.
	viewer_ = std::make_shared<Viewer>();
	viewer_->Start();
	viewer_->SetMethod(imageProcessor_->Method());
	imageProcessor_->SetFrontendCallback(
		[this](int curFrameId, const cv::Mat& curImg, const std::vector<cv::Point2f>& curKpts,
		       int refFrameId, const cv::Mat& refImg, const std::vector<cv::Point2f>& refPts,
		       const std::vector<cv::Point2f>& curPts,
		       int totalImages, int curImageIdx, int totalPairs, int curPairIdx,
		       double extractSec, double matchSec) {
			viewer_->SetFrontendData(curFrameId, curImg, curKpts, refFrameId, refImg, refPts, curPts,
			                         totalImages, curImageIdx, totalPairs, curPairIdx,
			                         extractSec, matchSec);
		});

	// Extract features from the images and match them across images
	imageProcessor_->ExtractAndMatchAll(imageDir);
	auto t1 = cv::getTickCount();

	// Frontend (feature/matching) displays are removed once backend reconstruction starts.
	viewer_->SetFrontendActive(false);

	reconstructor_->SetSceneUpdateCallback([this]() { UpdateViewerData(); });

	// Perform the 3D reconstruction using the extracted features and matches
	reconstructor_->Reconstruct(imageProcessor_->FramePtr(), imageProcessor_->MatchesPtr());
	auto t2 = cv::getTickCount();

	// Output the reconstructed point cloud and camera positions to a PLY file
	OutputPointCloud(workDir_, reconstructor_->SfmFeaturesPtr());
	auto t3 = cv::getTickCount();

	auto secondFactor = 1. / cv::getTickFrequency();
	std::cout << std::fixed << std::setprecision(2);
	std::cout << "Image processing time  : " << (t1 - t0) * secondFactor << " s\n";
	std::cout << "Reconstruction time    : " << (t2 - t1) * secondFactor << " s\n";
	std::cout << "Output points time     : " << (t3 - t2) * secondFactor << " s\n";

	// Make sure the final result is displayed, then wait for the user to close the window.
	UpdateViewerData();
	viewer_->Wait();
}

void System::UpdateViewerData() {
	if (!viewer_) {
		return;
	}
	const auto& sfmFeatures = *reconstructor_->SfmFeaturesPtr();
	const auto& frames = *imageProcessor_->FramePtr();
	const int curFrameId = reconstructor_->CurrentFrameId();

	std::vector<cv::Vec3d> pts;
	std::vector<cv::Vec3b> colors;
	std::vector<uchar> curVisible;
	pts.reserve(sfmFeatures.size());
	colors.reserve(sfmFeatures.size());
	curVisible.reserve(sfmFeatures.size());
	double threshold = cos(2 * CV_PI / 180);
	int registeredPoints = 0;
	for (const auto& fea : sfmFeatures) {
		if (fea.second.hasObject) {
			++registeredPoints;
		}
		if (fea.second.hasObject && fea.second.cosViewAngle < threshold && fea.second.features.size() >= 2) {
			cv::Point3f p = fea.second.Xw;
			pts.push_back(cv::Vec3d(p.x, p.y, p.z));
			colors.push_back(fea.second.features.front().color);
			bool visible = false;
			if (curFrameId >= 0) {
				for (const auto& obs : fea.second.features) {
					if (obs.imageId == curFrameId) {
						visible = true;
						break;
					}
				}
			}
			curVisible.push_back(visible ? 1 : 0);
		}
	}

	std::vector<cv::Vec3d> camPoses;
	std::vector<cv::Matx33d> camRots;
	camPoses.reserve(frames.size());
	camRots.reserve(frames.size());
	std::vector<cv::Vec3d> camIntrinsics;
	camIntrinsics.reserve(frames.size());
	std::vector<int> frameIdToCamIdx(frames.size(), -1);
	int currentCam = -1;
	for (size_t fi = 0; fi < frames.size(); ++fi) {
		const auto& frm = frames[fi];
		if (!frm.hasPose_) {
			continue;
		}
		const cv::Mat& R = frm.R_i_w_;
		const cv::Mat& t = frm.t_i_w_;
		cv::Vec3d tc(t.at<double>(0), t.at<double>(1), t.at<double>(2));
		cv::Vec3d c;
		c(0) = -(R.at<double>(0, 0) * tc(0) + R.at<double>(1, 0) * tc(1) + R.at<double>(2, 0) * tc(2));
		c(1) = -(R.at<double>(0, 1) * tc(0) + R.at<double>(1, 1) * tc(1) + R.at<double>(2, 1) * tc(2));
		c(2) = -(R.at<double>(0, 2) * tc(0) + R.at<double>(1, 2) * tc(1) + R.at<double>(2, 2) * tc(2));
		camPoses.push_back(c);
		frameIdToCamIdx[fi] = static_cast<int>(camPoses.size()) - 1;
		cv::Matx33d Rm;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				Rm(i, j) = R.at<double>(i, j);
			}
		}
		camRots.push_back(Rm);
		const cv::Mat& K = frm.K_;
		camIntrinsics.push_back(cv::Vec3d(K.at<double>(0, 0), frm.Width(), frm.Height()));
		if (static_cast<int>(fi) == curFrameId) {
			currentCam = static_cast<int>(camPoses.size()) - 1;
		}
	}

	// Build edges for cameras that share feature matches.
	std::vector<std::pair<int,int>> matchEdges;
	const auto& matches = *imageProcessor_->MatchesPtr();
	for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
		if (frameIdToCamIdx[i] < 0) {
			continue;
		}
		for (int j = 0; j < i; ++j) {
			if (!matches[i][j].empty() && frameIdToCamIdx[j] >= 0) {
				matchEdges.emplace_back(frameIdToCamIdx[i], frameIdToCamIdx[j]);
			}
		}
	}

	viewer_->SetData(pts, colors, camPoses, camRots, camIntrinsics, curVisible, currentCam, matchEdges);

	int totalImages = 0;
	int registeredImages = 0;
	for (const auto& frm : frames) {
		++totalImages;
		if (frm.hasPose_) {
			++registeredImages;
		}
	}
	const auto startPair = reconstructor_->StartPair();
	const bool shareIntrinsics = reconstructor_->ShareIntrinsicParams();
	const double currentFocal = shareIntrinsics ? frames.front().intrinsics_[0] : 0.0;
	viewer_->SetStatus(totalImages, registeredImages, registeredPoints,
	                   shareIntrinsics, currentFocal,
	                   reconstructor_->FocalEstimated(), reconstructor_->InitialFocal(),
	                   startPair.first, startPair.second,
	                   reconstructor_->PnpCount(), reconstructor_->EhCount(),
	                   imageProcessor_->Method());
}

}