#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sfm {

class Viewer {
public:
	Viewer();
	~Viewer();

	void SetData(const std::vector<cv::Vec3d>& pts,
	             const std::vector<cv::Vec3b>& colors,
	             const std::vector<cv::Vec3d>& camPoses,
	             const std::vector<cv::Matx33d>& camRots,
	             const std::vector<cv::Vec3d>& camIntrinsics,
	             const std::vector<uchar>& curVisible,
	             int currentCam,
	             const std::vector<std::pair<int,int>>& matchEdges = {});

	void SetStatus(int totalImages, int registeredImages, int pointCount,
	               bool shareIntrinsics, double focal,
	               bool focalEstimated, double initialFocal,
	               int initId1, int initId2,
	               int pnpCount, int ehCount,
	               const std::string& methodName = "");

	// Frontend (feature extraction / matching) display.
	// When active, the viewer renders the current frame with its feature points,
	// and during matching also the reference frame with match lines.
	void SetFrontendData(int curFrameId,
	                     const cv::Mat& curImg,
	                     const std::vector<cv::Point2f>& curKpts,
	                     int refFrameId,
	                     const cv::Mat& refImg,
	                     const std::vector<cv::Point2f>& refPts,
	                     const std::vector<cv::Point2f>& curPts,
	                     int totalImages, int curImageIdx,
	                     int totalPairs, int curPairIdx,
	                     double extractSec, double matchSec);

	// Enables/disables the frontend display (disabled when backend reconstruction starts).
	void SetFrontendActive(bool active);

	// Sets the feature extraction method shown in the overlay (shown before
	// SetStatus provides it, so the method is visible during extraction/matching).
	void SetMethod(const std::string& method);

	void Start();
	void Wait();
	void RequestClose();

private:
	void RenderLoop();
	void OnMouse(int event, int x, int y, int flags);
	static void MouseCallback(int event, int x, int y, int flags, void* userdata);

	void SnapshotData();
	void RenderFrame(cv::Mat& img);
	void RenderFrontend(cv::Mat& img);
	void Project(const cv::Vec3d& p, double& sx, double& sy, double& depth) const;
	void UpdateProjectionCache();
	void ComputeInitialScale();

	std::thread thread_;
	std::atomic<bool> running_{ false };

	std::mutex dataMutex_;
	std::vector<cv::Vec3d> pts_;
	std::vector<cv::Vec3b> colors_;
	std::vector<cv::Vec3d> camPoses_;
	std::vector<cv::Matx33d> camRots_;
	std::vector<cv::Vec3d> camIntrinsics_;
	std::vector<uchar> curVisible_;
	std::vector<std::pair<int,int>> matchEdges_;
	int currentCam_ = -1;
	int totalImages_ = 0;
	int registeredImages_ = 0;
	int pointCount_ = 0;
	bool shareIntrinsics_ = false;
	double currentFocal_ = 0.0;
	bool focalEstimated_ = false;
	double initialFocal_ = 0.0;
	int initId1_ = -1;
	int initId2_ = -1;
	int pnpCount_ = 0;
	int ehCount_ = 0;
	std::string methodName_;
	bool dataChanged_ = false;

	std::vector<cv::Vec3d> ptsCopy_;
	std::vector<cv::Vec3b> colorsCopy_;
	std::vector<cv::Vec3d> camPosesCopy_;
	std::vector<cv::Matx33d> camRotsCopy_;
	std::vector<cv::Vec3d> camIntrinsicsCopy_;
	std::vector<uchar> curVisibleCopy_;
	std::vector<std::pair<int,int>> matchEdgesCopy_;
	int currentCamCopy_ = -1;
	int totalImagesCopy_ = 0;
	int registeredImagesCopy_ = 0;
	int pointCountCopy_ = 0;
	bool shareIntrinsicsCopy_ = false;
	double currentFocalCopy_ = 0.0;
	bool focalEstimatedCopy_ = false;
	double initialFocalCopy_ = 0.0;
	int initId1Copy_ = -1;
	int initId2Copy_ = -1;
	int pnpCountCopy_ = 0;
	int ehCountCopy_ = 0;
	std::string methodNameCopy_;

	// Frontend display state (written by SetFrontendData, read by the render loop).
	bool frontendActive_ = false;
	int frontendCurFrameId_ = -1;
	cv::Mat frontendCurImg_;
	std::vector<cv::Point2f> frontendCurKpts_;
	int frontendRefFrameId_ = -1;
	cv::Mat frontendRefImg_;
	std::vector<cv::Point2f> frontendRefPts_;
	std::vector<cv::Point2f> frontendCurPts_;
	int frontendTotalImages_ = 0;
	int frontendCurImageIdx_ = 0;
	int frontendTotalPairs_ = 0;
	int frontendCurPairIdx_ = 0;
	double frontendExtractSec_ = 0.0;
	double frontendMatchSec_ = 0.0;

	// Extraction-phase summary, frozen once matching begins so the header stays
	// static while the matching info below updates in real time.
	bool frontendExtractSummaryValid_ = false;
	int frontendExtractImages_ = 0;
	int frontendExtractKpts_ = 0;

	// Snapshot copies of the frontend state.
	bool frontendActiveCopy_ = false;
	int frontendCurFrameIdCopy_ = -1;
	cv::Mat frontendCurImgCopy_;
	std::vector<cv::Point2f> frontendCurKptsCopy_;
	int frontendRefFrameIdCopy_ = -1;
	cv::Mat frontendRefImgCopy_;
	std::vector<cv::Point2f> frontendRefPtsCopy_;
	std::vector<cv::Point2f> frontendCurPtsCopy_;
	int frontendTotalImagesCopy_ = 0;
	int frontendCurImageIdxCopy_ = 0;
	int frontendTotalPairsCopy_ = 0;
	int frontendCurPairIdxCopy_ = 0;
	double frontendExtractSecCopy_ = 0.0;
	double frontendMatchSecCopy_ = 0.0;
	bool frontendExtractSummaryValidCopy_ = false;
	int frontendExtractImagesCopy_ = 0;
	int frontendExtractKptsCopy_ = 0;

	cv::Vec3d O_ = cv::Vec3d(0, 0, 0);
	cv::Matx33d R_ = cv::Matx33d::eye();
	double s_ = 1.0;
	cv::Vec3d T_ = cv::Vec3d(0, 0, 0);
	bool scaleInit_ = false;

	// Cached projection basis, refreshed once per frame in UpdateProjectionCache().
	cv::Vec3d projEye_;
	cv::Vec3d projFwd_;
	cv::Vec3d projRight_;
	cv::Vec3d projUp_;

	int width_ = 1440;
	int height_ = 960;
	double focal_ = 800.0;

	cv::Vec3d eye_ = cv::Vec3d(-10, 0, 0);
	cv::Vec3d viewRight_ = cv::Vec3d(0, -1, 0);
	cv::Vec3d viewUp_ = cv::Vec3d(0, 0, 1);
	cv::Vec3d viewFwd_ = cv::Vec3d(1, 0, 0);

	bool dragging_ = false;
	bool panMode_ = false;
	bool peripheral_ = false;
	int lastX_ = 0;
	int lastY_ = 0;
	double sceneRadius_ = 1.0;
	double camScale_ = 1.0;
};

}
