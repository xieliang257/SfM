#include "SuperPoint/SuperPoint.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace sfm {

namespace {

struct Point3 {
    float x = 0.f, y = 0.f;
    float s = 0.f;
};

void NMS8(const float* scoreMap, int H, int W,
          double confidenceThreshold, std::vector<Point3>& pts) {
    pts.clear();
    for (int cy = 0; cy < H; ++cy) {
        for (int cx = 0; cx < W; ++cx) {
            const float v = scoreMap[cy * W + cx];
            bool good1 = (v >= 0.1);
            bool good2 = (v >= 0.05);
            bool isMax = false;
            if (good2) {
                isMax = true;
                for (int dy = -1; dy <= 1 && isMax; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = cx + dx, ny = cy + dy;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H &&
                            scoreMap[ny * W + nx] >= v) {
                            isMax = false;
                            break;
                        }
                    }
                }
            }

            if (isMax || good1) {
                pts.push_back({static_cast<float>(cx), static_cast<float>(cy), v});
            }
        }
    }
}

float BilinearSample(const float* map, int C, int H, int W,
                     int channel, float y, float x) {
    y = std::max(0.f, std::min(y, static_cast<float>(H - 1)));
    x = std::max(0.f, std::min(x, static_cast<float>(W - 1)));
    const int y0 = std::min(static_cast<int>(std::floor(y)), H - 2);
    const int x0 = std::min(static_cast<int>(std::floor(x)), W - 2);
    const int y1 = y0 + 1, x1 = x0 + 1;
    const float dy = y - y0, dx = x - x0;
    const float* p = map + channel * H * W;
    return p[y0 * W + x0] * (1 - dy) * (1 - dx)
         + p[y0 * W + x1] * (1 - dy) * dx
         + p[y1 * W + x0] * dy * (1 - dx)
         + p[y1 * W + x1] * dy * dx;
}

}  // namespace

void SuperPoint::ortCheck(OrtStatus* s) {
    if (s) {
        std::cerr << "ONNX Runtime error: " << ort->GetErrorMessage(s) << std::endl;
        ort->ReleaseStatus(s);
        std::exit(1);
    }
}

SuperPoint::SuperPoint() = default;

SuperPoint::~SuperPoint() {
    if (inputName && allocator) allocator->Free(allocator, inputName);
    if (semiName && allocator) allocator->Free(allocator, semiName);
    if (descName && allocator) allocator->Free(allocator, descName);
    if (session) ort->ReleaseSession(session);
    if (memInfo) ort->ReleaseMemoryInfo(memInfo);
    if (env) ort->ReleaseEnv(env);
}

bool SuperPoint::Load(const std::string& modelPath) {
    // Get ORT API
    const OrtApiBase* apiBase = OrtGetApiBase();
    ort = apiBase->GetApi(ORT_API_VERSION);
    if (!ort) {
        std::cerr << "SuperPoint: failed to get ONNX Runtime API" << std::endl;
        return false;
    }

    // Create environment
    ortCheck(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "superpoint", &env));

    // Create session options with graph optimization
    OrtSessionOptions* sessOpt = nullptr;
    ortCheck(ort->CreateSessionOptions(&sessOpt));
    ortCheck(ort->SetSessionGraphOptimizationLevel(sessOpt, ORT_ENABLE_EXTENDED));

    // Create session from file path so ONNX Runtime can resolve
    // external-data files relative to the model directory.
    ortCheck(ort->CreateSession(env, modelPath.c_str(), sessOpt, &session));
    ort->ReleaseSessionOptions(sessOpt);

    // Get allocator and query input/output names
    ortCheck(ort->GetAllocatorWithDefaultOptions(&allocator));
    ortCheck(ort->SessionGetInputName(session, 0, allocator, &inputName));
    ortCheck(ort->SessionGetOutputName(session, 0, allocator, &semiName));
    ortCheck(ort->SessionGetOutputName(session, 1, allocator, &descName));

    // Create CPU memory info
    ortCheck(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo));

    loaded_ = true;
    std::cout << "SuperPoint: loaded ONNX model from " << modelPath << std::endl;
    return true;
}

bool SuperPoint::detectAndCompute(const cv::Mat& gray, int topK,
                                  std::vector<cv::KeyPoint>& kpts,
                                  cv::Mat& descriptors,
                                  std::vector<float>& scores,
                                  int nmsRadius,
                                  double confidenceThreshold,
                                  int maxDim) {
    kpts.clear();
    scores.clear();
    descriptors = cv::Mat();
    (void)nmsRadius;

    if (!loaded_ || gray.empty()) {
        return false;
    }

    // Convert to float [0,1]
    cv::Mat fimg;
    if (gray.type() == CV_8U) {
        gray.convertTo(fimg, CV_32F, 1.0 / 255.0);
    } else {
        fimg = gray.clone();
    }

    // Optionally downsample for speed
    double scale = 1.0;
    if (maxDim > 0) {
        int maxSide = std::max(fimg.cols, fimg.rows);
        if (maxSide > maxDim) {
            scale = static_cast<double>(maxDim) / maxSide;
            cv::Mat tmp;
            cv::resize(fimg, tmp, cv::Size(), scale, scale, cv::INTER_AREA);
            fimg = tmp;
        }
    }

    const int H = fimg.rows, W = fimg.cols;
    const int Hc = H / 8, Wc = W / 8;

    // Create input tensor (1, 1, H, W)
    const int64_t inputShape[] = {1, 1, H, W};
    const size_t inputSize = static_cast<size_t>(H) * W;
    OrtValue* inputTensor = nullptr;
    ortCheck(ort->CreateTensorWithDataAsOrtValue(
        memInfo, fimg.ptr<float>(), inputSize * sizeof(float),
        inputShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor));

    // Run inference
    const char* inputNames[] = {inputName};
    const char* outputNames[] = {semiName, descName};
    OrtValue* outputTensors[2] = {};
    ortCheck(ort->Run(session, nullptr, inputNames, &inputTensor, 1,
                      outputNames, 2, outputTensors));

    // Get output data pointers
    float* semiData = nullptr;
    float* descData = nullptr;
    ortCheck(ort->GetTensorMutableData(outputTensors[0], (void**)&semiData));
    ortCheck(ort->GetTensorMutableData(outputTensors[1], (void**)&descData));

    // Get output shapes
    OrtTensorTypeAndShapeInfo* semiInfo = nullptr;
    ortCheck(ort->GetTensorTypeAndShape(outputTensors[0], &semiInfo));
    std::vector<int64_t> semiDims(4);
    ortCheck(ort->GetDimensions(semiInfo, semiDims.data(), 4));
    const int64_t semiH = semiDims[2];
    const int64_t semiW = semiDims[3];
    ort->ReleaseTensorTypeAndShapeInfo(semiInfo);

    if (semiH != Hc || semiW != Wc) {
        ort->ReleaseValue(outputTensors[0]);
        ort->ReleaseValue(outputTensors[1]);
        ort->ReleaseValue(inputTensor);
        return false;
    }

    // Softmax over all 65 channels (including the dustbin channel), then use
    // the 64 sub-position channels.  The dustbin absorbs the "no keypoint"
    // probability, keeping background responses near zero.
    const int spSize = Hc * Wc;
    std::vector<float> conf(spSize * 64);
    for (int i = 0; i < spSize; ++i) {
        float maxVal = -1e30f;
        for (int c = 0; c < 65; ++c) {
            maxVal = std::max(maxVal, semiData[c * spSize + i]);
        }
        float sum = 0.f;
        for (int c = 0; c < 65; ++c) {
            sum += std::exp(semiData[c * spSize + i] - maxVal);
        }
        for (int c = 0; c < 64; ++c) {
            conf[c * spSize + i] =
                std::exp(semiData[c * spSize + i] - maxVal) / sum;
        }
    }

    // Coarse confidence map (Hc x Wc): max over the 64 sub-position channels.
    std::vector<float> confMap(spSize, -1e30f);
    for (int c = 0; c < 64; ++c) {
        const float* channel = conf.data() + c * spSize;
        for (int i = 0; i < spSize; ++i) {
            confMap[i] = std::max(confMap[i], channel[i]);
        }
    }

    // NMS: keep only cells strictly greater than all 8 neighbors.
    std::vector<Point3> pts;
    NMS8(confMap.data(), Hc, Wc, confidenceThreshold, pts);

    // Map coarse cell coordinates to full-resolution pixels (align_corners).
    const float yScale = Hc > 1 ? static_cast<float>(H - 1) / (Hc - 1) : 0.f;
    const float xScale = Wc > 1 ? static_cast<float>(W - 1) / (Wc - 1) : 0.f;
    for (auto& p : pts) {
        p.x *= xScale;
        p.y *= yScale;
    }

    // Remove border points (4px)
    const int border = 4;
    pts.erase(
        std::remove_if(pts.begin(), pts.end(), [W, H, border](const Point3& p) {
            return p.x < border || p.y < border ||
                   p.x >= W - border || p.y >= H - border;
        }),
        pts.end());

    // Top-K selection
    if ((int)pts.size() > topK) {
        std::partial_sort(pts.begin(), pts.begin() + topK, pts.end(),
                          [](const Point3& a, const Point3& b) { return a.s > b.s; });
        pts.resize(topK);
    }

    if (pts.empty()) {
        ort->ReleaseValue(outputTensors[0]);
        ort->ReleaseValue(outputTensors[1]);
        ort->ReleaseValue(inputTensor);
        return false;
    }

    // Sample descriptors at keypoint locations from the dense descriptor map.
    // grid_sample(align_corners=True): normalized coords (x/W*2-1) map to
    // src = x * (Wc-1) / W, matching the reference SuperPointFrontend.
    const int D = 256;
    const float kxScale = Wc > 1 ? static_cast<float>(Wc - 1) / W : 0.f;
    const float kyScale = Hc > 1 ? static_cast<float>(Hc - 1) / H : 0.f;
    kpts.reserve(pts.size());
    scores.reserve(pts.size());
    descriptors = cv::Mat(static_cast<int>(pts.size()), D, CV_32F);

    for (size_t i = 0; i < pts.size(); ++i) {
        const float kx = pts[i].x * kxScale;
        const float ky = pts[i].y * kyScale;

        float desc[256];
        float norm = 0.f;
        for (int d = 0; d < D; ++d) {
            desc[d] = BilinearSample(descData, D, Hc, Wc, d, ky, kx);
            norm += desc[d] * desc[d];
        }
        norm = std::sqrt(norm + 1e-8f);
        for (int d = 0; d < D; ++d) {
            desc[d] /= norm;
            descriptors.at<float>(static_cast<int>(i), d) = desc[d];
        }

        cv::KeyPoint kp;
        kp.pt.x = static_cast<float>(pts[i].x) / scale;
        kp.pt.y = static_cast<float>(pts[i].y) / scale;
        kp.response = pts[i].s;
        kpts.push_back(kp);
        scores.push_back(pts[i].s);
    }

    // Cleanup
    ort->ReleaseValue(outputTensors[0]);
    ort->ReleaseValue(outputTensors[1]);
    ort->ReleaseValue(inputTensor);

    return true;
}

}  // namespace sfm
