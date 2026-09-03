#include "Viewer/Viewer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace sfm {

namespace {
const char* kWindowName = "SFM Viewer";
const double kRotStep = 0.006;
const double kZoomStep = 0.1;
const double kNearPlane = 0.1;
}

Viewer::Viewer() {
	focal_ = std::min(width_, height_);
	R_<<0,1,0,0,0,-1,1,0,0;
}

Viewer::~Viewer() {
	RequestClose();
	Wait();
}

void Viewer::SetData(const std::vector<cv::Vec3d>& pts,
                     const std::vector<cv::Vec3b>& colors,
                     const std::vector<cv::Vec3d>& camPoses,
                     const std::vector<cv::Matx33d>& camRots,
                     const std::vector<cv::Vec3d>& camIntrinsics,
                     const std::vector<uchar>& curVisible,
                     int currentCam,
                     const std::vector<std::pair<int,int>>& matchEdges) {
	std::lock_guard<std::mutex> lock(dataMutex_);
	pts_ = pts;
	colors_ = colors;
	camPoses_ = camPoses;
	camRots_ = camRots;
	camIntrinsics_ = camIntrinsics;
	curVisible_ = curVisible;
	currentCam_ = currentCam;
	matchEdges_ = matchEdges;
	dataChanged_ = true;
}

void Viewer::SetStatus(int totalImages, int registeredImages, int pointCount,
                       bool shareIntrinsics, double focal,
                       bool focalEstimated, double initialFocal,
                       int initId1, int initId2,
                       int pnpCount, int ehCount,
                       const std::string& methodName) {
	std::lock_guard<std::mutex> lock(dataMutex_);
	totalImages_ = totalImages;
	registeredImages_ = registeredImages;
	pointCount_ = pointCount;
	shareIntrinsics_ = shareIntrinsics;
	currentFocal_ = focal;
	focalEstimated_ = focalEstimated;
	initialFocal_ = initialFocal;
	initId1_ = initId1;
	initId2_ = initId2;
	pnpCount_ = pnpCount;
	ehCount_ = ehCount;
	methodName_ = methodName;
	dataChanged_ = true;
}

void Viewer::SetFrontendData(int curFrameId,
                             const cv::Mat& curImg,
                             const std::vector<cv::Point2f>& curKpts,
                             int refFrameId,
                             const cv::Mat& refImg,
                             const std::vector<cv::Point2f>& refPts,
                             const std::vector<cv::Point2f>& curPts,
                             int totalImages, int curImageIdx,
                             int totalPairs, int curPairIdx,
                             double extractSec, double matchSec) {
	std::lock_guard<std::mutex> lock(dataMutex_);
	frontendCurFrameId_ = curFrameId;
	frontendCurImg_ = curImg;
	frontendCurKpts_ = curKpts;
	frontendRefFrameId_ = refFrameId;
	frontendRefImg_ = refImg;
	frontendRefPts_ = refPts;
	frontendCurPts_ = curPts;
	frontendTotalImages_ = totalImages;
	frontendCurImageIdx_ = curImageIdx;
	frontendTotalPairs_ = totalPairs;
	frontendCurPairIdx_ = curPairIdx;
	frontendExtractSec_ = extractSec;
	frontendMatchSec_ = matchSec;
	frontendActive_ = true;
	// During the extraction phase (no reference frame yet) keep refreshing the
	// extraction summary; once matching starts the header stays frozen on it.
	if (refFrameId < 0) {
		frontendExtractSummaryValid_ = true;
		frontendExtractImages_ = totalImages;
		frontendExtractKpts_ = static_cast<int>(curKpts.size());
	}
	dataChanged_ = true;
}

void Viewer::SetFrontendActive(bool active) {
	std::lock_guard<std::mutex> lock(dataMutex_);
	frontendActive_ = active;
	// Keep the last feature/matching image data: it stays visible as an overlay
	// once the backend reconstruction view takes over.
	dataChanged_ = true;
}

void Viewer::SetMethod(const std::string& method) {
	std::lock_guard<std::mutex> lock(dataMutex_);
	methodName_ = method;
	dataChanged_ = true;
}

void Viewer::Start() {
	if (running_) {
		return;
	}
	running_ = true;
	thread_ = std::thread(&Viewer::RenderLoop, this);
}

void Viewer::Wait() {
	if (thread_.joinable()) {
		thread_.join();
	}
}

void Viewer::RequestClose() {
	running_ = false;
}

void Viewer::MouseCallback(int event, int x, int y, int flags, void* userdata) {
	static_cast<Viewer*>(userdata)->OnMouse(event, x, y, flags);
}

void Viewer::OnMouse(int event, int x, int y, int flags) {
	const int cx = width_ / 2;
	const int cy = height_ / 2;

	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		dragging_ = true;
		panMode_ = false;
		peripheral_ = (std::abs(x - cx) > width_ * 0.35 || std::abs(y - cy) > height_ * 0.35);
		lastX_ = x;
		lastY_ = y;
		break;
	case cv::EVENT_MBUTTONDOWN:
	case cv::EVENT_RBUTTONDOWN:
		dragging_ = true;
		panMode_ = true;
		lastX_ = x;
		lastY_ = y;
		break;
	case cv::EVENT_LBUTTONUP:
	case cv::EVENT_MBUTTONUP:
	case cv::EVENT_RBUTTONUP:
		dragging_ = false;
		break;
	case cv::EVENT_MOUSEMOVE:
		if (!dragging_) {
			break;
		}
		if (panMode_) {
			const double d = std::max(O_(0) - eye_(0), kNearPlane);
			const double k = d / focal_;
			const double dx = static_cast<double>(x - lastX_);
			const double dy = static_cast<double>(y - lastY_);
			const cv::Vec3d right = R_ * viewRight_;
			const cv::Vec3d up = R_ * viewUp_;
			T_ += (right * dx - up * dy) * k;
		}
		else if (peripheral_) {
			double a1 = std::atan2(lastY_ - cy, lastX_ - cx);
			double a2 = std::atan2(y - cy, x - cx);
			double da = a2 - a1;
			if (da > CV_PI) da -= 2.0 * CV_PI;
			if (da < -CV_PI) da += 2.0 * CV_PI;
			const double c = std::cos(-da), s = std::sin(-da);
			cv::Matx33d Rx(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
			R_ = R_ * Rx;
		}
		else {
			const double thz = -(x - lastX_) * kRotStep;
			const double thy = (y - lastY_) * kRotStep;
			const double cz = std::cos(thz), sz = std::sin(thz);
			const double cy = std::cos(thy), sy = std::sin(thy);
			cv::Matx33d Rz(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0);
			cv::Matx33d Ry(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
			R_ = R_ * Rz * Ry;
		}
		lastX_ = x;
		lastY_ = y;
		break;
	case cv::EVENT_MOUSEWHEEL: {
		const int delta = cv::getMouseWheelDelta(flags);
		const double m = std::exp(-delta * kZoomStep);
		// With Shift held, the wheel resizes the displayed cameras instead of zooming.
		if (flags & cv::EVENT_FLAG_SHIFTKEY) {
			camScale_ *= (1.0 / m);
			camScale_ = std::max(0.01, std::min(camScale_, 100.0));
			break;
		}
		eye_ = eye_ * m;
		const double len = cv::norm(eye_);
		if (len < 0.5) eye_ *= (0.5 / len);
		if (len > 1e4) eye_ *= (1e4 / len);
		break;
	}
	default:
		break;
	}
}

void Viewer::SnapshotData() {
	std::lock_guard<std::mutex> lock(dataMutex_);
	if (!dataChanged_) {
		return;
	}
	ptsCopy_ = pts_;
	colorsCopy_ = colors_;
	camPosesCopy_ = camPoses_;
	camRotsCopy_ = camRots_;
	camIntrinsicsCopy_ = camIntrinsics_;
	curVisibleCopy_ = curVisible_;
	matchEdgesCopy_ = matchEdges_;
	currentCamCopy_ = currentCam_;
	totalImagesCopy_ = totalImages_;
	registeredImagesCopy_ = registeredImages_;
	pointCountCopy_ = pointCount_;
	shareIntrinsicsCopy_ = shareIntrinsics_;
	currentFocalCopy_ = currentFocal_;
	focalEstimatedCopy_ = focalEstimated_;
	initialFocalCopy_ = initialFocal_;
	initId1Copy_ = initId1_;
	initId2Copy_ = initId2_;
	pnpCountCopy_ = pnpCount_;
	ehCountCopy_ = ehCount_;
	methodNameCopy_ = methodName_;
	frontendActiveCopy_ = frontendActive_;
	frontendCurFrameIdCopy_ = frontendCurFrameId_;
	frontendCurImgCopy_ = frontendCurImg_;
	frontendCurKptsCopy_ = frontendCurKpts_;
	frontendRefFrameIdCopy_ = frontendRefFrameId_;
	frontendRefImgCopy_ = frontendRefImg_;
	frontendRefPtsCopy_ = frontendRefPts_;
	frontendCurPtsCopy_ = frontendCurPts_;
	frontendTotalImagesCopy_ = frontendTotalImages_;
	frontendCurImageIdxCopy_ = frontendCurImageIdx_;
	frontendTotalPairsCopy_ = frontendTotalPairs_;
	frontendCurPairIdxCopy_ = frontendCurPairIdx_;
	frontendExtractSecCopy_ = frontendExtractSec_;
	frontendMatchSecCopy_ = frontendMatchSec_;
	frontendExtractSummaryValidCopy_ = frontendExtractSummaryValid_;
	frontendExtractImagesCopy_ = frontendExtractImages_;
	frontendExtractKptsCopy_ = frontendExtractKpts_;
	dataChanged_ = false;
}

void Viewer::UpdateProjectionCache() {
	projEye_ = R_ * eye_;
	projFwd_ = R_ * viewFwd_;
	projRight_ = R_ * viewRight_;
	projUp_ = R_ * viewUp_;
}

void Viewer::Project(const cv::Vec3d& p, double& sx, double& sy, double& depth) const {
	const cv::Vec3d pe = s_ * (p - O_) + O_ + T_;
	const cv::Vec3d v = pe - projEye_;
	depth = v.dot(projFwd_);
	sx = width_ / 2.0 + focal_ * v.dot(projRight_) / depth;
	sy = height_ / 2.0 - focal_ * v.dot(projUp_) / depth;
}

void Viewer::ComputeInitialScale() {
	UpdateProjectionCache();
	std::vector<cv::Vec3d> scenePts = ptsCopy_;
	if (scenePts.empty()) {
		scenePts = camPosesCopy_;
	}
	if (scenePts.empty()) {
		scaleInit_ = true;
		return;
	}
	double minX = 1e30, maxX = -1e30, minY = 1e30, maxY = -1e30, maxR = 0.0;
	for (const auto& p : scenePts) {
		double sx, sy, d;
		Project(p, sx, sy, d);
		if (d <= kNearPlane) {
			continue;
		}
		minX = std::min(minX, sx);
		maxX = std::max(maxX, sx);
		minY = std::min(minY, sy);
		maxY = std::max(maxY, sy);
		maxR = std::max(maxR, cv::norm(p - O_));
	}
	sceneRadius_ = std::max(maxR, 1e-6);
	const double extX = maxX - minX, extY = maxY - minY;
	if (extX > 1e-6 && extY > 1e-6) {
		const double fit = 0.8 * std::min(width_, height_) / std::max(extX, extY);
		s_ = std::max(1e-6, std::min(fit, 1e6));
	}
	scaleInit_ = true;
}

void Viewer::RenderFrontend(cv::Mat& img) {
	// Same background as the reconstruction view.
	img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(255, 255, 255));

	const cv::Mat& cur = frontendCurImgCopy_;
	const cv::Mat& ref = frontendRefImgCopy_;
	const bool hasRef = !ref.empty() && frontendRefFrameIdCopy_ >= 0;

	const int margin = 16;

	struct PlacedImage {
		cv::Rect rect;
		double scale;
	};

	// Resizes an image and places it flush to the top-right of the given area.
	auto place = [&](const cv::Mat& src, const cv::Rect& area) -> PlacedImage {
		const int srcW = std::max(1, src.cols), srcH = std::max(1, src.rows);
		double s = std::min(double(area.width) / srcW, double(area.height) / srcH);
		int w = std::max(1, (int)std::round(srcW * s));
		int h = std::max(1, (int)std::round(srcH * s));
		cv::Rect rect(area.x + (area.width - w), area.y, w, h);
		cv::Mat dst;
		cv::resize(src, dst, cv::Size(w, h), 0, 0, cv::INTER_AREA);
		dst.copyTo(img(rect));
		return { rect, s };
	};

	auto drawKpts = [&](const PlacedImage& p, const std::vector<cv::Point2f>& kpts, const cv::Scalar& color) {
		for (const auto& k : kpts) {
			cv::Point pt(cvRound(p.rect.x + k.x * p.scale), cvRound(p.rect.y + k.y * p.scale));
			if (p.rect.contains(pt)) {
				cv::circle(img, pt, 2, color, -1);
			}
		}
	};

	if (hasRef) {
		// Matching phase: both images side by side, flush to the top.
		const int halfW = (width_ - margin * 3) / 2;
		const int refW = width_ - margin * 3 - halfW;
		PlacedImage pCur = place(cur, cv::Rect(margin, 0, halfW, height_));
		PlacedImage pRef = place(ref, cv::Rect(margin * 2 + halfW, 0, refW, height_));

		const size_t n = std::min(frontendCurPtsCopy_.size(), frontendRefPtsCopy_.size());
		for (size_t i = 0; i < n; ++i) {
			cv::Point a(cvRound(pCur.rect.x + frontendCurPtsCopy_[i].x * pCur.scale),
			            cvRound(pCur.rect.y + frontendCurPtsCopy_[i].y * pCur.scale));
			cv::Point b(cvRound(pRef.rect.x + frontendRefPtsCopy_[i].x * pRef.scale),
			            cvRound(pRef.rect.y + frontendRefPtsCopy_[i].y * pRef.scale));
			cv::line(img, a, b, cv::Scalar(0, 180, 0), 1);
			cv::circle(img, a, 2, cv::Scalar(0, 0, 255), -1);
			cv::circle(img, b, 2, cv::Scalar(255, 0, 0), -1);
		}
	}
	else {
		// Feature extraction phase: current frame flush to the top-right.
		PlacedImage pCur = place(cur, cv::Rect(0, 0, width_ - margin, height_));
		drawKpts(pCur, frontendCurKptsCopy_, cv::Scalar(0, 0, 255));
	}

	// Progress text overlaid at the top-left, styled like the reconstruction view.
	const int headerLeft = 12;
	const int lineH = 22;
	int headerY = 28;
	char buf[256];

	// Extraction summary always on top; during matching it stays frozen.
	std::snprintf(buf, sizeof(buf), "Feature method: %s", methodNameCopy_.empty() ? "n/a" : methodNameCopy_.c_str());
	cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
	            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	headerY += lineH;

	if (hasRef) {
		// Frozen extraction summary from when extraction finished.
		std::snprintf(buf, sizeof(buf), "Images: %d / %d",
		              frontendExtractImagesCopy_, frontendExtractImagesCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
		headerY += lineH;
		std::snprintf(buf, sizeof(buf), "Keypoints: %d   Extract %.2f s",
		              frontendExtractKptsCopy_, frontendExtractSecCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);

		// Separator, then the live matching info on its own.
		headerY += lineH;
		cv::line(img, cv::Point(headerLeft, headerY - 8),
		         cv::Point(headerLeft + 500, headerY - 8), cv::Scalar(130, 130, 130), 1);
		headerY += 8;

		std::snprintf(buf, sizeof(buf), "Pairs: %d / %d",
		              frontendCurPairIdxCopy_, frontendTotalPairsCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
		headerY += lineH;
		std::snprintf(buf, sizeof(buf), "Match points: %d   Match %.2f s",
		              (int)std::min(frontendCurPtsCopy_.size(), frontendRefPtsCopy_.size()),
		              frontendMatchSecCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	}
	else {
		// Extraction phase: live progress.
		std::snprintf(buf, sizeof(buf), "Images: %d / %d",
		              frontendCurImageIdxCopy_, frontendTotalImagesCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
		headerY += lineH;
		std::snprintf(buf, sizeof(buf), "Keypoints: %d   Extract %.2f s",
		              (int)frontendCurKptsCopy_.size(), frontendExtractSecCopy_);
		cv::putText(img, buf, cv::Point(headerLeft, headerY), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	}
}

void Viewer::RenderFrame(cv::Mat& img) {
	if (frontendActiveCopy_) {
		RenderFrontend(img);
		return;
	}
	img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(255, 255, 255));
	UpdateProjectionCache();

	// Draw commands collected by the projection workers, rasterized serially afterwards
	// so the draw order (and therefore occlusion) matches the naive single-threaded loop.
	struct DrawCmd {
		int x, y;
		size_t ci;
	};

	const int kPointSize = 3;
	auto rasterize = [&](int x, int y, size_t ci) {
		// Points seen by the current frame are overlaid in red.
		const cv::Vec3b c = (ci < curVisibleCopy_.size() && curVisibleCopy_[ci])
		                        ? cv::Vec3b(0, 0, 255)
		                        : colorsCopy_[ci];
		for (int dy = 0; dy < kPointSize; ++dy) {
			if (y + dy < 0 || y + dy >= height_) {
				continue;
			}
			cv::Vec3b* row = img.ptr<cv::Vec3b>(y + dy);
			for (int dx = 0; dx < kPointSize; ++dx) {
				if (x + dx < 0 || x + dx >= width_) {
					continue;
				}
				row[x + dx] = c;
			}
		}
	};

	const size_t nPts = ptsCopy_.size();
	unsigned hw = std::thread::hardware_concurrency();
	if (hw == 0) hw = 1;
	const unsigned nThreads = std::max(1u, std::min(hw, static_cast<unsigned>((nPts + 262143u) / 262144u)));

	if (nThreads == 1) {
		for (size_t i = 0; i < nPts; ++i) {
			double sx, sy, d;
			Project(ptsCopy_[i], sx, sy, d);
			if (d <= kNearPlane) {
				continue;
			}
			const int x = cvRound(sx), y = cvRound(sy);
			if (x < 0 || y < 0 || x >= width_ || y >= height_) {
				continue;
			}
			rasterize(x, y, i);
		}
	}
	else {
		std::vector<std::vector<DrawCmd>> lists(nThreads);
		std::vector<std::thread> threads;
		const size_t stride = (nPts + nThreads - 1) / nThreads;
		for (unsigned t = 0; t < nThreads; ++t) {
			threads.emplace_back([&, t]() {
				auto& list = lists[t];
				const size_t start = t * stride;
				const size_t end = std::min(start + stride, nPts);
				list.reserve(end - start);
				for (size_t i = start; i < end; ++i) {
					double sx, sy, d;
					Project(ptsCopy_[i], sx, sy, d);
					if (d <= kNearPlane) {
						continue;
					}
					const int x = cvRound(sx), y = cvRound(sy);
					if (x < 0 || y < 0 || x >= width_ || y >= height_) {
						continue;
					}
					list.push_back(DrawCmd{ x, y, i });
				}
			});
		}
		for (auto& th : threads) {
			th.join();
		}
		for (const auto& list : lists) {
			for (const auto& cmd : list) {
				rasterize(cmd.x, cmd.y, cmd.ci);
			}
		}
	}

	auto drawSegment = [&](const cv::Vec3d& a, const cv::Vec3d& b, const cv::Scalar& color, int thickness = 1) {
		double ax, ay, ad, bx, by, bd;
		Project(a, ax, ay, ad);
		Project(b, bx, by, bd);
		if (ad <= kNearPlane || bd <= kNearPlane) {
			return;
		}
		cv::line(img, cv::Point(cvRound(ax), cvRound(ay)), cv::Point(cvRound(bx), cvRound(by)), color, thickness);
	};

	auto projFixed = [&](const cv::Vec3d& p, double& sx, double& sy, double& depth) {
		const cv::Vec3d v = p - projEye_;
		depth = v.dot(projFwd_);
		sx = width_ / 2.0 + focal_ * v.dot(projRight_) / depth;
		sy = height_ / 2.0 - focal_ * v.dot(projUp_) / depth;
	};

	const cv::Vec3d origin(0, 0, 0);
	// Constant on-screen length (pixels) so the axes do not grow with zoom.
	const double kAxisPx = 70.0;
	const cv::Vec3d axisDirs[3] = { cv::Vec3d(1, 0, 0),
	                                cv::Vec3d(0, 1, 0),
	                                cv::Vec3d(0, 0, 1) };
	const cv::Scalar axisCols[3] = { cv::Scalar(0, 0, 255),
	                                 cv::Scalar(0, 255, 0),
	                                 cv::Scalar(255, 0, 0) };
	const char* axisLabels[3] = { "x", "y", "z" };
	double ox, oy, od;
	projFixed(origin, ox, oy, od);
	for (int i = 0; i < 3; ++i) {
		if (od <= kNearPlane) {
			continue;
		}
		double tx, ty, td;
		projFixed(origin + axisDirs[i], tx, ty, td);
		if (td <= kNearPlane) {
			continue;
		}
		// Screen direction of the world axis (project a unit increment), then
		// scale to a fixed pixel length.
		double dx = tx - ox, dy = ty - oy;
		const double len = std::sqrt(dx * dx + dy * dy);
		if (len < 1e-6) {
			continue;   // axis points toward/away from the camera
		}
		dx /= len;
		dy /= len;
		const int ex = cvRound(ox + dx * kAxisPx);
		const int ey = cvRound(oy + dy * kAxisPx);
		cv::line(img, cv::Point(cvRound(ox), cvRound(oy)),
		         cv::Point(ex, ey), axisCols[i], 2);
		cv::putText(img, axisLabels[i], cv::Point(ex + 6, ey - 6),
		            cv::FONT_HERSHEY_SIMPLEX, 0.6, axisCols[i], 2, cv::LINE_AA);
	}
	{
		if (od > kNearPlane) {
			cv::circle(img, cv::Point(cvRound(ox), cvRound(oy)), 4,
			           cv::Scalar(0, 0, 255), -1);
			cv::putText(img, "O", cv::Point(cvRound(ox) - 14, cvRound(oy) - 10),
			            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
		}
	}

	const double camLen = sceneRadius_ * 0.3 * camScale_;
	for (size_t i = 0; i < camPosesCopy_.size(); ++i) {
		const cv::Vec3d& c = camPosesCopy_[i];
		const cv::Matx33d& Rc = camRotsCopy_[i];
		cv::Vec3d fwd(Rc(2, 0), Rc(2, 1), Rc(2, 2));
		cv::Vec3d right(Rc(0, 0), Rc(0, 1), Rc(0, 2));
		cv::Vec3d up(-Rc(1, 0), -Rc(1, 1), -Rc(1, 2));

		// Real horizontal / vertical half-FOV tangents from the camera intrinsics.
		const double f = i < camIntrinsicsCopy_.size() ? camIntrinsicsCopy_[i](0) : 0.0;
		const double w = i < camIntrinsicsCopy_.size() ? camIntrinsicsCopy_[i](1) : 0.0;
		const double h = i < camIntrinsicsCopy_.size() ? camIntrinsicsCopy_[i](2) : 0.0;
		const double hw = (f > 1e-6 && w > 1e-6) ? w / (2.0 * f) : std::tan(30.0 * CV_PI / 180.0);
		const double hh = (f > 1e-6 && h > 1e-6) ? h / (2.0 * f) : hw;

		// The four image-plane corner rays in cyclic order (bottom-left, bottom-right,
		// top-right, top-left); scale them to a display size.
		std::vector<cv::Vec3d> corner(4);
		for (int k = 0; k < 4; ++k) {
			const double sx = (k == 1 || k == 2) ? 1.0 : -1.0;
			const double sy = (k >= 2) ? 1.0 : -1.0;
			const cv::Vec3d ray = fwd + (right * sx * hw + up * sy * hh);
			corner[k] = c + ray * (camLen / cv::norm(ray));
		}

		// 8 lines: 4 from the camera center to the corners + 4 connecting the corners.
		// The current frame's camera is drawn thicker.
		const int lw = (static_cast<int>(i) == currentCamCopy_) ? 3 : 1;
		for (int k = 0; k < 4; ++k) {
			drawSegment(c, corner[k], cv::Scalar(0, 0, 255), lw);
			drawSegment(corner[k], corner[(k + 1) % 4], cv::Scalar(0, 0, 255), lw);
		}
	}

	// Draw lines between cameras that have feature matches.
	for (const auto& e : matchEdgesCopy_) {
		if (e.first < 0 || e.first >= static_cast<int>(camPosesCopy_.size()) ||
		    e.second < 0 || e.second >= static_cast<int>(camPosesCopy_.size())) {
			continue;
		}
		drawSegment(camPosesCopy_[e.first], camPosesCopy_[e.second],
		            cv::Scalar(0, 0, 255), 1);
	}

	cv::putText(img, "L: rotate | M/R: pan | wheel: zoom | Shift+wheel: cam size | q/ESC: quit",
	            cv::Point(12, height_ - 14), cv::FONT_HERSHEY_SIMPLEX, 0.5,
	            cv::Scalar(70, 70, 70), 1, cv::LINE_AA);

	char buf[256];
	int y = 28;

	// Feature method first, matching the frontend overlay.
	std::snprintf(buf, sizeof(buf), "Feature method: %s", methodNameCopy_.empty() ? "n/a" : methodNameCopy_.c_str());
	cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
	            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	y += 22;

	// Frontend (feature/matching) summary text from the last step.
	if (!frontendCurImgCopy_.empty()) {
		std::snprintf(buf, sizeof(buf), "Extract images: %d    Extract time: %.2f s",
		              frontendTotalImagesCopy_, frontendExtractSecCopy_);
		cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
		y += 22;
		if (frontendRefFrameIdCopy_ >= 0) {
			std::snprintf(buf, sizeof(buf), "Match pairs: %d    Match time: %.2f s",
			              frontendTotalPairsCopy_, frontendMatchSecCopy_);
		}
		else {
			std::snprintf(buf, sizeof(buf), "Match pairs: 0    Match time: 0.00 s");
		}
		cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
		            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
		y += 22;
	}

	// Backend reconstruction summary text.
	std::snprintf(buf, sizeof(buf), "Images: %d / %d    Registered points: %d",
	              registeredImagesCopy_, totalImagesCopy_, pointCountCopy_);
	cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
	            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	y += 22;
	std::snprintf(buf, sizeof(buf), "Init frames: %d + %d    Growth: PnP %d, E/H %d",
	              initId1Copy_, initId2Copy_, pnpCountCopy_, ehCountCopy_);
	cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
	            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
	y += 22;
	if (shareIntrinsicsCopy_) {
		if (focalEstimatedCopy_) {
			std::snprintf(buf, sizeof(buf), "Intrinsics: shared    Estimated initial f: %.1f    Optimized f: %.1f",
			              initialFocalCopy_, currentFocalCopy_);
		}
		else {
			std::snprintf(buf, sizeof(buf), "Intrinsics: shared    Optimized f: %.1f", currentFocalCopy_);
		}
	}
	else {
		if (focalEstimatedCopy_) {
			std::snprintf(buf, sizeof(buf), "Intrinsics: not shared    Estimated initial f: %.1f",
			              initialFocalCopy_);
		}
		else {
			std::snprintf(buf, sizeof(buf), "Intrinsics: not shared");
		}
	}
	cv::putText(img, buf, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.55,
	            cv::Scalar(40, 40, 40), 1, cv::LINE_8);
}

void Viewer::RenderLoop() {
	cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(kWindowName, Viewer::MouseCallback, this);

	cv::Mat img;
	while (running_) {
		SnapshotData();
		if (!scaleInit_ && !frontendActiveCopy_) {
			ComputeInitialScale();
		}
		RenderFrame(img);
		cv::imshow(kWindowName, img);
		const int key = cv::waitKey(15);
		if (key == 'q' || key == 27) {
			break;
		}
	}
	running_ = false;
	cv::destroyWindow(kWindowName);
}

}
