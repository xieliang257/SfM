#include "XFeat/XFeat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>

namespace sfm {

namespace {

constexpr float kBnEps = 1e-5f;

// cubic kernel used by torch grid_sample bicubic (a = -0.75)
float Cubic(float t) {
    const float a = -0.75f;
    const float at = std::fabs(t);
    if (at <= 1.f) {
        return ((a + 2.f) * at - (a + 3.f)) * at * at + 1.f;
    }
    if (at < 2.f) {
        return ((a * at - 5.f * a) * at + 8.f * a) * at - 4.f * a;
    }
    return 0.f;
}

// torch grid_sample coordinate, align_corners = false
inline float GridCoord(float pos, int fullSize, int mapSize) {
    // grid = 2 * pos / (fullSize - 1) - 1
    // ix   = ((grid + 1) * mapSize - 1) / 2
    return pos * mapSize / (fullSize - 1.f) - 0.5f;
}

}  // namespace

XFeat::XFeat() = default;

XFeat::~XFeat() {
    if (inputName && allocator) allocator->Free(allocator, inputName);
    if (featsName && allocator) allocator->Free(allocator, featsName);
    if (keypointsName && allocator) allocator->Free(allocator, keypointsName);
    if (heatmapName && allocator) allocator->Free(allocator, heatmapName);
    if (session && ort) ort->ReleaseSession(session);
    if (memInfo && ort) ort->ReleaseMemoryInfo(memInfo);
    if (env && ort) ort->ReleaseEnv(env);
}

bool XFeat::Load(const std::string& weightPath) {
    std::ifstream in(weightPath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "XFeat: failed to open weight file: " << weightPath << std::endl;
        return false;
    }

    int32_t nTensors = 0;
    in.read(reinterpret_cast<char*>(&nTensors), sizeof(int32_t));
    if (!in.good() || nTensors <= 0 || nTensors > 100000) {
        std::cerr << "XFeat: invalid weight file header: " << weightPath << std::endl;
        return false;
    }

    struct Entry {
        std::string name;
        int64_t offset = 0;   // byte offset of data within weights_
        std::vector<int> dims;
    };
    std::vector<Entry> entries;
    std::map<std::string, size_t> nameIndex;

    weights_.clear();
    for (int i = 0; i < nTensors; ++i) {
        int32_t nameLen = 0;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(int32_t));
        if (!in.good() || nameLen <= 0 || nameLen > 1024) {
            return false;
        }
        Entry e;
        e.name.resize(nameLen);
        in.read(&e.name[0], nameLen);
        int32_t ndim = 0;
        in.read(reinterpret_cast<char*>(&ndim), sizeof(int32_t));
        if (!in.good() || ndim <= 0 || ndim > 8) {
            return false;
        }
        int64_t nElem = 1;
        for (int d = 0; d < ndim; ++d) {
            int32_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
            e.dims.push_back(dim);
            nElem *= dim;
        }
        e.offset = weights_.size();
        weights_.resize(weights_.size() + nElem);
        in.read(reinterpret_cast<char*>(weights_.data() + e.offset), nElem * sizeof(float));
        if (!in.good()) {
            return false;
        }
        nameIndex[e.name] = entries.size();
        entries.push_back(std::move(e));
    }

    auto Get = [&](const std::string& name) -> const float* {
        auto it = nameIndex.find(name);
        if (it == nameIndex.end()) {
            std::cerr << "XFeat: missing weight tensor: " << name << std::endl;
            return nullptr;
        }
        return weights_.data() + entries[it->second].offset;
    };

    // Bind all layer tensors.
    w_skip1 = Get("skip1.1.weight");  b_skip1 = Get("skip1.1.bias");
    for (int i = 0; i < 4; ++i) {
        w_b1[i] = Get("block1." + std::to_string(i) + ".layer.0.weight");
        s_b1[i] = Get("block1." + std::to_string(i) + ".layer.1.scale");
        h_b1[i] = Get("block1." + std::to_string(i) + ".layer.1.shift");
    }
    for (int i = 0; i < 2; ++i) {
        w_b2[i] = Get("block2." + std::to_string(i) + ".layer.0.weight");
        s_b2[i] = Get("block2." + std::to_string(i) + ".layer.1.scale");
        h_b2[i] = Get("block2." + std::to_string(i) + ".layer.1.shift");
    }
    for (int i = 0; i < 3; ++i) {
        w_b3[i] = Get("block3." + std::to_string(i) + ".layer.0.weight");
        s_b3[i] = Get("block3." + std::to_string(i) + ".layer.1.scale");
        h_b3[i] = Get("block3." + std::to_string(i) + ".layer.1.shift");
    }
    for (int i = 0; i < 3; ++i) {
        w_b4[i] = Get("block4." + std::to_string(i) + ".layer.0.weight");
        s_b4[i] = Get("block4." + std::to_string(i) + ".layer.1.scale");
        h_b4[i] = Get("block4." + std::to_string(i) + ".layer.1.shift");
    }
    for (int i = 0; i < 4; ++i) {
        w_b5[i] = Get("block5." + std::to_string(i) + ".layer.0.weight");
        s_b5[i] = Get("block5." + std::to_string(i) + ".layer.1.scale");
        h_b5[i] = Get("block5." + std::to_string(i) + ".layer.1.shift");
    }
    w_bf[0] = Get("block_fusion.0.layer.0.weight");
    w_bf[1] = Get("block_fusion.1.layer.0.weight");
    w_bf[2] = Get("block_fusion.2.weight");
    s_bf[0] = Get("block_fusion.0.layer.1.scale");  h_bf[0] = Get("block_fusion.0.layer.1.shift");
    s_bf[1] = Get("block_fusion.1.layer.1.scale");  h_bf[1] = Get("block_fusion.1.layer.1.shift");
    b_bf2 = Get("block_fusion.2.bias");

    for (int i = 0; i < 2; ++i) {
        w_hh[i] = Get("heatmap_head." + std::to_string(i) + ".layer.0.weight");
        s_hh[i] = Get("heatmap_head." + std::to_string(i) + ".layer.1.scale");
        h_hh[i] = Get("heatmap_head." + std::to_string(i) + ".layer.1.shift");
    }
    w_hh[2] = Get("heatmap_head.2.weight");
    b_hh2 = Get("heatmap_head.2.bias");

    for (int i = 0; i < 3; ++i) {
        w_kh[i] = Get("keypoint_head." + std::to_string(i) + ".layer.0.weight");
        s_kh[i] = Get("keypoint_head." + std::to_string(i) + ".layer.1.scale");
        h_kh[i] = Get("keypoint_head." + std::to_string(i) + ".layer.1.shift");
    }
    w_kh[3] = Get("keypoint_head.3.weight");
    b_kh3 = Get("keypoint_head.3.bias");

    const float* all[] = { w_skip1, b_skip1,
        w_b1[0], w_b1[1], w_b1[2], w_b1[3],
        w_b2[0], w_b2[1], w_b3[0], w_b3[1], w_b3[2],
        w_b4[0], w_b4[1], w_b4[2], w_b5[0], w_b5[1], w_b5[2], w_b5[3],
        w_bf[0], w_bf[1], w_bf[2], b_bf2,
        w_hh[0], w_hh[1], w_hh[2], b_hh2,
        w_kh[0], w_kh[1], w_kh[2], w_kh[3], b_kh3,
        s_b1[0], s_b1[1], s_b1[2], s_b1[3], h_b1[0], h_b1[1], h_b1[2], h_b1[3],
        s_b2[0], s_b2[1], h_b2[0], h_b2[1],
        s_b3[0], s_b3[1], s_b3[2], h_b3[0], h_b3[1], h_b3[2],
        s_b4[0], s_b4[1], s_b4[2], h_b4[0], h_b4[1], h_b4[2],
        s_b5[0], s_b5[1], s_b5[2], s_b5[3], h_b5[0], h_b5[1], h_b5[2], h_b5[3],
        s_bf[0], s_bf[1], h_bf[0], h_bf[1],
        s_hh[0], s_hh[1], h_hh[0], h_hh[1],
        s_kh[0], s_kh[1], s_kh[2], h_kh[0], h_kh[1], h_kh[2] };
    for (const float* p : all) {
        if (p == nullptr) {
            std::cerr << "XFeat: incomplete weight file" << std::endl;
            return false;
        }
    }

    loaded_ = true;
    return true;
}

void XFeat::ortCheck(OrtStatus* s) {
    if (s) {
        std::cerr << "XFeat ONNX Runtime error: " << ort->GetErrorMessage(s) << std::endl;
        ort->ReleaseStatus(s);
        std::exit(1);
    }
}

bool XFeat::LoadONNX(const std::string& modelPath) {
    const OrtApiBase* apiBase = OrtGetApiBase();
    ort = apiBase->GetApi(ORT_API_VERSION);
    if (!ort) {
        std::cerr << "XFeat: failed to get ONNX Runtime API" << std::endl;
        return false;
    }

    ortCheck(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xfeat", &env));

    OrtSessionOptions* sessOpt = nullptr;
    ortCheck(ort->CreateSessionOptions(&sessOpt));
    ortCheck(ort->SetSessionGraphOptimizationLevel(sessOpt, ORT_ENABLE_EXTENDED));
    ortCheck(ort->CreateSession(env, modelPath.c_str(), sessOpt, &session));
    ort->ReleaseSessionOptions(sessOpt);

    ortCheck(ort->GetAllocatorWithDefaultOptions(&allocator));
    ortCheck(ort->SessionGetInputName(session, 0, allocator, &inputName));
    ortCheck(ort->SessionGetOutputName(session, 0, allocator, &featsName));
    ortCheck(ort->SessionGetOutputName(session, 1, allocator, &keypointsName));
    ortCheck(ort->SessionGetOutputName(session, 2, allocator, &heatmapName));
    ortCheck(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo));

    onnxLoaded_ = true;
    loaded_ = true;
    std::cout << "XFeat: loaded ONNX model from " << modelPath << std::endl;
    return true;
}

bool XFeat::ForwardONNX(const cv::Mat& gray, int H, int W,
                        Tensor4& feats, Tensor4& keypoints, Tensor4& heatmap) {
    if (!onnxLoaded_ || H <= 0 || W <= 0) {
        return false;
    }

    // Build the (1,1,H,W) float32 input in [0,1].
    std::vector<float> buf;
    if (gray.type() == CV_32F) {
        buf.assign(gray.ptr<float>(), gray.ptr<float>() + size_t(H) * W);
    } else {
        buf.resize(size_t(H) * W);
        const uchar* src = gray.data;
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = src[i] * (1.f / 255.f);
        }
    }

    const int64_t inputShape[] = {1, 1, H, W};
    OrtValue* inputTensor = nullptr;
    ortCheck(ort->CreateTensorWithDataAsOrtValue(
        memInfo, buf.data(), buf.size() * sizeof(float),
        inputShape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor));

    const char* inNames[] = {inputName};
    const char* outNames[] = {featsName, keypointsName, heatmapName};
    OrtValue* outTensors[3] = {};
    ortCheck(ort->Run(session, nullptr, inNames, &inputTensor, 1, outNames, 3, outTensors));

    float* outData[3] = {};
    for (int i = 0; i < 3; ++i) {
        ortCheck(ort->GetTensorMutableData(outTensors[i], (void**)&outData[i]));
    }

    // Output shapes are [1, C, H/8, W/8] (model resizes to a multiple of 32).
    int64_t dims[4];
    OrtTensorTypeAndShapeInfo* info = nullptr;
    ortCheck(ort->GetTensorTypeAndShape(outTensors[0], &info));
    ortCheck(ort->GetDimensions(info, dims, 4));
    ort->ReleaseTensorTypeAndShapeInfo(info);

    feats.Resize((int)dims[1], (int)dims[2], (int)dims[3]);
    memcpy(feats.d.data(), outData[0], feats.d.size() * sizeof(float));

    ortCheck(ort->GetTensorTypeAndShape(outTensors[1], &info));
    ortCheck(ort->GetDimensions(info, dims, 4));
    ort->ReleaseTensorTypeAndShapeInfo(info);
    keypoints.Resize((int)dims[1], (int)dims[2], (int)dims[3]);
    memcpy(keypoints.d.data(), outData[1], keypoints.d.size() * sizeof(float));

    ortCheck(ort->GetTensorTypeAndShape(outTensors[2], &info));
    ortCheck(ort->GetDimensions(info, dims, 4));
    ort->ReleaseTensorTypeAndShapeInfo(info);
    heatmap.Resize((int)dims[1], (int)dims[2], (int)dims[3]);
    memcpy(heatmap.d.data(), outData[2], heatmap.d.size() * sizeof(float));

    ort->ReleaseValue(outTensors[0]);
    ort->ReleaseValue(outTensors[1]);
    ort->ReleaseValue(outTensors[2]);
    ort->ReleaseValue(inputTensor);
    return true;
}

// ---------------------------------------------------------------- CNN ops

void XFeat::Conv(const Tensor4& in, const float* weight, const float* bias,
                 bool hasBias, int outC, int k, int stride, int pad, Tensor4& out) {
    Tensor4 tmp;
    const Tensor4* src = &in;
    if (&out == &in) {
        tmp = in;  // in-place call: keep a copy before out.Resize() frees it
        src = &tmp;
    }
    const int inC = src->C, H = src->H, W = src->W;
    const int outH = (H + 2 * pad - k) / stride + 1;
    const int outW = (W + 2 * pad - k) / stride + 1;
    out.Resize(outC, outH, outW);

    const int k2 = k * k;
    const int numThreads = std::max(1, (int)std::thread::hardware_concurrency());

    auto Work = [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; ++oc) {
            const float* w = weight + (size_t)oc * inC * k2;
            const float biasVal = hasBias ? bias[oc] : 0.f;
            float* obase = out.d.data() + (size_t)oc * outH * outW;
            for (int oy = 0; oy < outH; ++oy) {
                const int iy0 = oy * stride - pad;
                const int kyLo = std::max(0, -iy0);
                const int kyHi = std::min(k - 1, H - 1 - iy0);
                float* orow = obase + (size_t)oy * outW;
                for (int ox = 0; ox < outW; ++ox) {
                    const int ix0 = ox * stride - pad;
                    const int kxLo = std::max(0, -ix0);
                    const int kxHi = std::min(k - 1, W - 1 - ix0);
                    float acc = biasVal;
                    for (int ic = 0; ic < inC; ++ic) {
                        const float* ip = src->Ptr(ic, 0, 0);
                        const float* wp = w + (size_t)ic * k2;
                        for (int ky = kyLo; ky <= kyHi; ++ky) {
                            const float* irow = ip + (size_t)(iy0 + ky) * W;
                            const float* wrow = wp + (size_t)ky * k;
                            for (int kx = kxLo; kx <= kxHi; ++kx) {
                                acc += irow[ix0 + kx] * wrow[kx];
                            }
                        }
                    }
                    orow[ox] = acc;
                }
            }
        }
    };

    if (numThreads <= 1 || outC <= 4) {
        Work(0, outC);
    } else {
        std::vector<std::thread> threads;
        int chunk = (outC + numThreads - 1) / numThreads;
        for (int oc = 0; oc < outC; oc += chunk) {
            threads.emplace_back(Work, oc, std::min(oc + chunk, outC));
        }
        for (auto& t : threads) t.join();
    }
}

void XFeat::Conv1x1(const Tensor4& in, const float* weight, const float* bias,
                    bool hasBias, int outC, Tensor4& out) {
    Tensor4 tmp;
    const Tensor4* src = &in;
    if (&out == &in) {
        tmp = in;
        src = &tmp;
    }
    const int inC = src->C, H = src->H, W = src->W;
    out.Resize(outC, H, W);
    const int numThreads = std::max(1, (int)std::thread::hardware_concurrency());

    auto Work = [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; ++oc) {
            const float* w = weight + (size_t)oc * inC;
            const float biasVal = hasBias ? bias[oc] : 0.f;
            float* obase = out.d.data() + (size_t)oc * H * W;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float acc = biasVal;
                    for (int ic = 0; ic < inC; ++ic) {
                        acc += w[ic] * src->Ptr(ic, y, x)[0];
                    }
                    obase[(size_t)y * W + x] = acc;
                }
            }
        }
    };

    if (numThreads <= 1 || outC <= 4) {
        Work(0, outC);
    } else {
        std::vector<std::thread> threads;
        int chunk = (outC + numThreads - 1) / numThreads;
        for (int oc = 0; oc < outC; oc += chunk) {
            threads.emplace_back(Work, oc, std::min(oc + chunk, outC));
        }
        for (auto& t : threads) t.join();
    }
}

void XFeat::BatchNorm(const Tensor4& in, const float* scale, const float* shift, Tensor4& out) {
    Tensor4 tmp;
    const Tensor4* src = &in;
    if (&out == &in) {
        tmp = in;
        src = &tmp;
    }
    out.Resize(src->C, src->H, src->W);
    const size_t hw = (size_t)src->H * src->W;
    for (int c = 0; c < src->C; ++c) {
        const float s = scale[c], t = shift[c];
        const float* ip = src->Ptr(c, 0, 0);
        float* op = out.Ptr(c, 0, 0);
        for (size_t i = 0; i < hw; ++i) {
            op[i] = ip[i] * s + t;
        }
    }
}

void XFeat::Relu(Tensor4& x) {
    for (float& v : x.d) {
        if (v < 0.f) v = 0.f;
    }
}

void XFeat::InstanceNorm1(Tensor4& x) {
    double sum = 0.0, sum2 = 0.0;
    for (float v : x.d) {
        sum += v;
        sum2 += (double)v * v;
    }
    const double n = (double)x.d.size();
    const double mean = sum / n;
    const double var = std::max(0.0, sum2 / n - mean * mean);
    const float invStd = (float)(1.0 / std::sqrt(var + kBnEps));
    for (float& v : x.d) {
        v = (v - (float)mean) * invStd;
    }
}

void XFeat::AvgPool4(const Tensor4& in, Tensor4& out) {
    const int outH = in.H / 4, outW = in.W / 4;
    out.Resize(in.C, outH, outW);
    for (int c = 0; c < in.C; ++c) {
        for (int oy = 0; oy < outH; ++oy) {
            for (int ox = 0; ox < outW; ++ox) {
                float acc = 0.f;
                for (int dy = 0; dy < 4; ++dy) {
                    const float* ip = in.Ptr(c, oy * 4 + dy, ox * 4);
                    acc += ip[0] + ip[1] + ip[2] + ip[3];
                }
                *out.Ptr(c, oy, ox) = acc * 0.0625f;
            }
        }
    }
}

void XFeat::ResizeBilinear(const Tensor4& in, int outH, int outW, Tensor4& out) {
    // torch F.interpolate(mode='bilinear', align_corners=False)
    out.Resize(in.C, outH, outW);
    const double scaleY = (double)in.H / outH;
    const double scaleX = (double)in.W / outW;
    for (int c = 0; c < in.C; ++c) {
        for (int oy = 0; oy < outH; ++oy) {
            const double sy = (oy + 0.5) * scaleY - 0.5;
            const double syc = std::max(0.0, sy);
            const int y0 = (int)std::floor(syc);
            const int y1 = std::min(y0 + 1, in.H - 1);
            const double wy = syc - y0;
            for (int ox = 0; ox < outW; ++ox) {
                const double sx = (ox + 0.5) * scaleX - 0.5;
                const double sxc = std::max(0.0, sx);
                const int x0 = (int)std::floor(sxc);
                const int x1 = std::min(x0 + 1, in.W - 1);
                const double wx = sxc - x0;
                const float v00 = *in.Ptr(c, y0, x0), v01 = *in.Ptr(c, y0, x1);
                const float v10 = *in.Ptr(c, y1, x0), v11 = *in.Ptr(c, y1, x1);
                const double top = v00 * (1 - wx) + v01 * wx;
                const double bot = v10 * (1 - wx) + v11 * wx;
                *out.Ptr(c, oy, ox) = (float)(top * (1 - wy) + bot * wy);
            }
        }
    }
}

void XFeat::Add3(const Tensor4& a, const Tensor4& b, const Tensor4& c, Tensor4& out) {
    out.Resize(a.C, a.H, a.W);
    for (size_t i = 0; i < out.d.size(); ++i) {
        out.d[i] = a.d[i] + b.d[i] + c.d[i];
    }
}

void XFeat::Unfold8(const Tensor4& in, Tensor4& out) {
    // in: (1, 1, H, W)  ->  out: (64, H/8, W/8), channel index = hh * 8 + ww
    const int outH = in.H / 8, outW = in.W / 8;
    out.Resize(64, outH, outW);
    for (int hh = 0; hh < 8; ++hh) {
        for (int ww = 0; ww < 8; ++ww) {
            const int ch = hh * 8 + ww;
            for (int oy = 0; oy < outH; ++oy) {
                const float* ip = in.Ptr(0, oy * 8 + hh, ww);
                float* op = out.Ptr(ch, oy, 0);
                for (int ox = 0; ox < outW; ++ox) {
                    op[ox] = ip[(size_t)ox * 8];
                }
            }
        }
    }
}

// -------------------------------------------------------------- forward

void XFeat::Forward(const cv::Mat& gray, int H, int W,
                    Tensor4& feats, Tensor4& keypoints, Tensor4& heatmap) {
    std::cout << "hand write forward\n";
    // Input tensor (1, 1, H, W).
    Tensor4 x;
    x.Resize(1, H, W);
    {
        const uchar* src = gray.data;
        float* dst = x.d.data();
        if (gray.type() == CV_32F) {
            const float* srcf = (const float*)gray.data;
            for (int i = 0; i < H * W; ++i) dst[i] = srcf[i];
        } else {
            for (int i = 0; i < H * W; ++i) dst[i] = src[i] * (1.f / 255.f);
        }
    }
    // Downsample to a multiple of 32 using torch-compatible bilinear
    // (F.interpolate mode='bilinear', align_corners=False).
    Tensor4 xr;
    ResizeBilinear(x, (H / 32) * 32, (W / 32) * 32, xr);
    x = std::move(xr);
    InstanceNorm1(x);

    // skip connection: avg pool 4 + 1x1 conv (1 -> 24)
    Tensor4 skip, x1;
    AvgPool4(x, skip);
    Conv1x1(skip, w_skip1, b_skip1, true, 24, skip);

    // block1
    Conv(x, w_b1[0], nullptr, false, 4, 3, 1, 1, x1);
    BatchNorm(x1, s_b1[0], h_b1[0], x1); Relu(x1);
    Conv(x1, w_b1[1], nullptr, false, 8, 3, 2, 1, x1);
    BatchNorm(x1, s_b1[1], h_b1[1], x1); Relu(x1);
    Conv(x1, w_b1[2], nullptr, false, 8, 3, 1, 1, x1);
    BatchNorm(x1, s_b1[2], h_b1[2], x1); Relu(x1);
    Conv(x1, w_b1[3], nullptr, false, 24, 3, 2, 1, x1);
    BatchNorm(x1, s_b1[3], h_b1[3], x1); Relu(x1);

    // x2 = block2(x1 + skip)
    Tensor4 x2;
    x2.C = x1.C; x2.H = x1.H; x2.W = x1.W;
    x2.d.resize(x1.d.size());
    for (size_t i = 0; i < x1.d.size(); ++i) x2.d[i] = x1.d[i] + skip.d[i];
    for (int i = 0; i < 2; ++i) {
        Conv(x2, w_b2[i], nullptr, false, 24, 3, 1, 1, x2);
        BatchNorm(x2, s_b2[i], h_b2[i], x2);
        Relu(x2);
    }

    // block3 -> x3 at 1/8
    Tensor4 x3;
    Conv(x2, w_b3[0], nullptr, false, 64, 3, 2, 1, x3);
    BatchNorm(x3, s_b3[0], h_b3[0], x3); Relu(x3);
    Conv(x3, w_b3[1], nullptr, false, 64, 3, 1, 1, x3);
    BatchNorm(x3, s_b3[1], h_b3[1], x3); Relu(x3);
    Conv(x3, w_b3[2], nullptr, false, 64, 1, 1, 0, x3);
    BatchNorm(x3, s_b3[2], h_b3[2], x3); Relu(x3);

    // block4 -> x4 at 1/16
    Tensor4 x4;
    Conv(x3, w_b4[0], nullptr, false, 64, 3, 2, 1, x4);
    BatchNorm(x4, s_b4[0], h_b4[0], x4); Relu(x4);
    Conv(x4, w_b4[1], nullptr, false, 64, 3, 1, 1, x4);
    BatchNorm(x4, s_b4[1], h_b4[1], x4); Relu(x4);
    Conv(x4, w_b4[2], nullptr, false, 64, 3, 1, 1, x4);
    BatchNorm(x4, s_b4[2], h_b4[2], x4); Relu(x4);

    // block5 -> x5 at 1/32
    Tensor4 x5;
    Conv(x4, w_b5[0], nullptr, false, 128, 3, 2, 1, x5);
    BatchNorm(x5, s_b5[0], h_b5[0], x5); Relu(x5);
    Conv(x5, w_b5[1], nullptr, false, 128, 3, 1, 1, x5);
    BatchNorm(x5, s_b5[1], h_b5[1], x5); Relu(x5);
    Conv(x5, w_b5[2], nullptr, false, 128, 3, 1, 1, x5);
    BatchNorm(x5, s_b5[2], h_b5[2], x5); Relu(x5);
    Conv(x5, w_b5[3], nullptr, false, 64, 1, 1, 0, x5);
    BatchNorm(x5, s_b5[3], h_b5[3], x5); Relu(x5);

    // pyramid fusion: x3 + up(x4) + up(x5)
    Tensor4 x4r, x5r, fused;
    ResizeBilinear(x4, x3.H, x3.W, x4r);
    ResizeBilinear(x5, x3.H, x3.W, x5r);
    Add3(x3, x4r, x5r, fused);

    // block_fusion
    Tensor4 fx;
    Conv(fused, w_bf[0], nullptr, false, 64, 3, 1, 1, fx);
    BatchNorm(fx, s_bf[0], h_bf[0], fx); Relu(fx);
    Conv(fx, w_bf[1], nullptr, false, 64, 3, 1, 1, fx);
    BatchNorm(fx, s_bf[1], h_bf[1], fx); Relu(fx);
    Conv1x1(fx, w_bf[2], b_bf2, true, 64, feats);

    // heatmap head
    Tensor4 h;
    Conv1x1(feats, w_hh[0], nullptr, false, 64, h);
    BatchNorm(h, s_hh[0], h_hh[0], h); Relu(h);
    Conv1x1(h, w_hh[1], nullptr, false, 64, h);
    BatchNorm(h, s_hh[1], h_hh[1], h); Relu(h);
    Conv1x1(h, w_hh[2], b_hh2, true, 1, heatmap);
    for (float& v : heatmap.d) v = 1.f / (1.f + std::exp(-v));  // sigmoid

    // keypoint head on the 8x8-unfolded normalized input
    Tensor4 u;
    Unfold8(x, u);
    for (int i = 0; i < 3; ++i) {
        Conv1x1(u, w_kh[i], nullptr, false, 64, u);
        BatchNorm(u, s_kh[i], h_kh[i], u);
        Relu(u);
    }
    Conv1x1(u, w_kh[3], b_kh3, true, 65, keypoints);
}

// ------------------------------------------------- post-processing

void XFeat::BuildHeatmap(const Tensor4& keypoints, cv::Mat& heatmap) {
    const int CH = keypoints.H, CW = keypoints.W;
    const int H = CH * 8, W = CW * 8;
    heatmap = cv::Mat(H, W, CV_32F);
    for (int py = 0; py < CH; ++py) {
        for (int px = 0; px < CW; ++px) {
            // softmax over the 65 logit channels
            float maxv = keypoints.Ptr(0, py, px)[0];
            for (int c = 1; c < 65; ++c) {
                maxv = std::max(maxv, keypoints.Ptr(c, py, px)[0]);
            }
            float sum = 0.f;
            for (int c = 0; c < 65; ++c) {
                sum += std::exp(keypoints.Ptr(c, py, px)[0] - maxv);
            }
            const float inv = 1.f / sum;
            for (int c = 0; c < 64; ++c) {
                const int hh = c / 8, ww = c % 8;
                heatmap.at<float>(py * 8 + hh, px * 8 + ww) =
                    std::exp(keypoints.Ptr(c, py, px)[0] - maxv) * inv;
            }
        }
    }
}

void XFeat::NMS(const cv::Mat& heatmap, float threshold, int kernel,
                std::vector<cv::Point2i>& pts) {
    const int H = heatmap.rows, W = heatmap.cols;
    const int pad = kernel / 2;
    cv::Mat localMax = cv::Mat(H, W, CV_32F);
    for (int y = 0; y < H; ++y) {
        const int yLo = std::max(0, y - pad), yHi = std::min(H - 1, y + pad);
        float* lm = localMax.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const int xLo = std::max(0, x - pad), xHi = std::min(W - 1, x + pad);
            float best = -1e30f;
            for (int yy = yLo; yy <= yHi; ++yy) {
                const float* row = heatmap.ptr<float>(yy);
                for (int xx = xLo; xx <= xHi; ++xx) {
                    best = std::max(best, row[xx]);
                }
            }
            lm[x] = best;
        }
    }
    // Row-major scan, matching torch nonzero() ordering.
    for (int y = 0; y < H; ++y) {
        const float* hm = heatmap.ptr<float>(y);
        const float* lm = localMax.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            if (hm[x] == lm[x] && hm[x] > threshold) {
                pts.emplace_back(x, y);
            }
        }
    }
}

void XFeat::SampleBilinear(const Tensor4& coarse, const std::vector<cv::Point2i>& fullPts,
                           int fullH, int fullW, std::vector<float>& vals) {
    vals.resize(fullPts.size());
    const int CH = coarse.H, CW = coarse.W;
    for (size_t i = 0; i < fullPts.size(); ++i) {
        const float gx = GridCoord((float)fullPts[i].x, fullW, CW);
        const float gy = GridCoord((float)fullPts[i].y, fullH, CH);
        const int x0 = (int)std::floor(gx);
        const int x1 = std::min(x0 + 1, CW - 1);
        const float wx = gx - x0;
        const int y0 = (int)std::floor(gy);
        const int y1 = std::min(y0 + 1, CH - 1);
        const float wy = gy - y0;
        // single-channel reliability map
        const float v00 = *coarse.Ptr(0, std::max(0, y0), std::max(0, x0));
        const float v01 = *coarse.Ptr(0, std::max(0, y0), x1);
        const float v10 = *coarse.Ptr(0, y1, std::max(0, x0));
        const float v11 = *coarse.Ptr(0, y1, x1);
        vals[i] = (v00 * (1 - wx) + v01 * wx) * (1 - wy) + (v10 * (1 - wx) + v11 * wx) * wy;
    }
}

void XFeat::SampleBicubic(const Tensor4& coarse, const std::vector<cv::Point2i>& fullPts,
                          int fullH, int fullW, cv::Mat& descriptors) {
    const int C = coarse.C, CH = coarse.H, CW = coarse.W;
    const size_t n = fullPts.size();
    descriptors.create((int)n, C, CV_32F);
    for (size_t i = 0; i < n; ++i) {
        const float gx = GridCoord((float)fullPts[i].x, fullW, CW);
        const float gy = GridCoord((float)fullPts[i].y, fullH, CH);
        const int x0 = (int)std::floor(gx);
        const int y0 = (int)std::floor(gy);
        float cx[4], cy[4];
        for (int n = -1; n <= 2; ++n) {
            cx[n + 1] = Cubic(gx - (x0 + n));
            cy[n + 1] = Cubic(gy - (y0 + n));
        }
        float* desc = descriptors.ptr<float>((int)i);
        for (int c = 0; c < C; ++c) {
            float val = 0.f;
            for (int dy = -1; dy <= 2; ++dy) {
                const int sy = std::min(std::max(y0 + dy, 0), CH - 1);
                for (int dx = -1; dx <= 2; ++dx) {
                    const int sx = std::min(std::max(x0 + dx, 0), CW - 1);
                    val += cy[dy + 1] * cx[dx + 1] * *coarse.Ptr(c, sy, sx);
                }
            }
            desc[c] = val;
        }
    }
}

// ----------------------------------------------------------- public API

bool XFeat::detectAndCompute(const cv::Mat& gray, int topK,
                             std::vector<cv::KeyPoint>& kpts,
                             cv::Mat& descriptors,
                             std::vector<float>& scores) {
    if (!loaded_ || gray.empty()) {
        return false;
    }

    const int H = gray.rows, W = gray.cols;
    const int _H = (H / 32) * 32, _W = (W / 32) * 32;
    if (_H == 0 || _W == 0) {
        return false;
    }
    const float rh = (float)H / _H, rw = (float)W / _W;

    Tensor4 feats, keypoints, heatmap;
    if (onnxLoaded_) {
        if (!ForwardONNX(gray, H, W, feats, keypoints, heatmap)) {
            return false;
        }
    } else {
        Forward(gray, H, W, feats, keypoints, heatmap);
    }

    // L2-normalize the descriptor map over channels.
    {
        const size_t hw = (size_t)feats.H * feats.W;
        for (size_t yx = 0; yx < hw; ++yx) {
            float acc = 0.f;
            for (int c = 0; c < 64; ++c) {
                const float v = feats.Ptr(c, 0, 0)[yx];
                acc += v * v;
            }
            const float inv = 1.f / (std::sqrt(acc) + 1e-12f);
            for (int c = 0; c < 64; ++c) {
                feats.Ptr(c, 0, 0)[yx] *= inv;
            }
        }
    }

    cv::Mat heatmapFull;
    BuildHeatmap(keypoints, heatmapFull);

    std::vector<cv::Point2i> pts;
    NMS(heatmapFull, 0.05f, 5, pts);

    if (pts.empty()) {
        return false;
    }

    // Reliability scores: nearest(heatmap) * bilinear(reliability map).
    std::vector<float> scoresAll(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        scoresAll[i] = heatmapFull.at<float>(pts[i].y, pts[i].x);
    }
    std::vector<float> rel;
    SampleBilinear(heatmap, pts, _H, _W, rel);
    for (size_t i = 0; i < pts.size(); ++i) {
        scoresAll[i] *= rel[i];
    }
    for (size_t i = 0; i < pts.size(); ++i) {
        if (pts[i].x == 0 && pts[i].y == 0) {
            scoresAll[i] = -1.f;
        }
    }

    // Sort by descending score, keep top-k.
    std::vector<size_t> idx(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return scoresAll[a] > scoresAll[b];
    });
    const size_t keep = std::min((size_t)topK, pts.size());
    std::vector<cv::Point2i> kept(keep);
    std::vector<float> keptScores(keep);
    for (size_t i = 0; i < keep; ++i) {
        kept[i] = pts[idx[i]];
        keptScores[i] = scoresAll[idx[i]];
    }

    // Bicubic descriptor sampling at the coarse map.
    cv::Mat desc;
    SampleBicubic(feats, kept, _H, _W, desc);

    // L2-normalize descriptors and apply the scale factors.
    kpts.clear();
    scores.clear();
    for (size_t i = 0; i < keep; ++i) {
        if (!(keptScores[i] > 0.f)) {
            continue;
        }
        float* d = desc.ptr<float>((int)i);
        float acc = 0.f;
        for (int c = 0; c < 64; ++c) acc += d[c] * d[c];
        const float inv = 1.f / (std::sqrt(acc) + 1e-12f);
        for (int c = 0; c < 64; ++c) d[c] *= inv;

        cv::KeyPoint kp;
        kp.pt.x = kept[i].x * rw;
        kp.pt.y = kept[i].y * rh;
        kp.response = keptScores[i];
        kp.size = 8.f;
        kp.class_id = -1;
        kpts.push_back(kp);
        scores.push_back(keptScores[i]);
    }

    if (kpts.empty()) {
        return false;
    }
    descriptors = desc.clone();
    // Keep only rows that survived the score filter.
    {
        cv::Mat filtered(0, 64, CV_32F);
        int keptRow = 0;
        for (size_t i = 0; i < keep; ++i) {
            if (!(keptScores[i] > 0.f)) continue;
            filtered.push_back(descriptors.row((int)i));
            ++keptRow;
        }
        descriptors = filtered;
    }
    return true;
}

}  // namespace sfm
