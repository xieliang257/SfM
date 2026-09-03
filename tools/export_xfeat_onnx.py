# -*- coding: utf-8 -*-
"""
Export the XFeat (CVPR 2024) feature-extraction backbone to a single ONNX file
that is numerically identical to the hand-written C++ forward pass in
src/XFeat/XFeat.cc.

Weights are read from the compact binary file produced by
tools/export_xfeat_weights.py (weights/xfeat.bin), so the exported graph uses
the exact same tensors as the C++ engine. The graph mirrors XFeat::Forward()
step by step:

    input  :  images [1, 1, H, W]  (grayscale float32 in [0, 1], any size)
    1. F.interpolate bilinear align_corners=False to (H//32*32, W//32*32)
    2. instance norm over the single channel (eps = 1e-5)
    3. CNN backbone (skip, block1..block5, pyramid fusion, block_fusion)
    4. heads
    outputs:
        feats     [1, 64, H/8, W/8]   dense L2-feature map (descriptors)
        keypoints [1, 65, H/8, W/8]   keypoint logit map
        heatmap   [1,  1, H/8, W/8]   reliability map (sigmoid)

Sparse post-processing (softmax over the 65 logits, NMS, top-k, descriptor
sampling) is intentionally kept in C++ and shared by both the hand-written and
the ONNX Runtime path.

Usage:
    python3 tools/export_xfeat_onnx.py --bin weights/xfeat.bin \
        --output weights/xfeat.onnx
"""

import argparse
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_bin(path):
    tensors = {}
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            ln = struct.unpack("<i", f.read(4))[0]
            name = f.read(ln).decode("latin-1")
            nd = struct.unpack("<i", f.read(4))[0]
            dims = struct.unpack("<%di" % nd, f.read(4 * nd))
            nelem = 1
            for d in dims:
                nelem *= d
            data = np.frombuffer(f.read(4 * nelem), dtype="<f4").reshape(list(dims))
            tensors[name] = torch.from_numpy(data)
    return tensors


class XFeatOnnx(nn.Module):
    """Torch replica of XFeat::Forward() from src/XFeat/XFeat.cc."""

    def __init__(self, t):
        super().__init__()
        self.t = t
        for name, tensor in t.items():
            self.register_buffer("t_" + name.replace(".", "_"), tensor)

    # -- helpers ---------------------------------------------------------
    def bn(self, x, block, i):
        # pre-folded BatchNorm: out = in * scale + shift
        s = self.t[block + "." + str(i) + ".layer.1.scale"].view(1, -1, 1, 1)
        h = self.t[block + "." + str(i) + ".layer.1.shift"].view(1, -1, 1, 1)
        return x * s + h

    def conv3(self, x, w, bias=None, stride=1, pad=1):
        return F.conv2d(x, w, bias, stride=stride, padding=pad)

    def conv1(self, x, w, bias=None):
        return F.conv2d(x, w, bias, stride=1, padding=0)

    def conv_bn_relu(self, x, block, i, k, stride, pad, out_c):
        y = F.conv2d(x, self.t[block + "." + str(i) + ".layer.0.weight"],
                     None, stride=stride, padding=pad)
        return F.relu(self.bn(y, block, i))

    # -- network ---------------------------------------------------------
    def forward(self, x):
        # 1) resize to a multiple of 32 (F.interpolate align_corners=False)
        h, w = x.shape[2], x.shape[3]
        hc, wc = (h // 32) * 32, (w // 32) * 32
        x = F.interpolate(x, size=(hc, wc), mode="bilinear", align_corners=False)

        # 2) instance norm over the single channel (eps = 1e-5)
        x = F.instance_norm(x, eps=1e-5)

        # skip connection: avg pool 4 + 1x1 conv (1 -> 24)
        skip = F.avg_pool2d(x, kernel_size=4, stride=4)
        skip = self.conv1(skip, self.t["skip1.1.weight"], self.t["skip1.1.bias"])

        # block1
        x1 = self.conv_bn_relu(x, "block1", 0, 3, 1, 1, 4)
        x1 = self.conv_bn_relu(x1, "block1", 1, 3, 2, 1, 8)
        x1 = self.conv_bn_relu(x1, "block1", 2, 3, 1, 1, 8)
        x1 = self.conv_bn_relu(x1, "block1", 3, 3, 2, 1, 24)

        # x2 = block2(x1 + skip)
        x2 = x1 + skip
        x2 = self.conv_bn_relu(x2, "block2", 0, 3, 1, 1, 24)
        x2 = self.conv_bn_relu(x2, "block2", 1, 3, 1, 1, 24)

        # block3 -> x3 at 1/8
        x3 = self.conv_bn_relu(x2, "block3", 0, 3, 2, 1, 64)
        x3 = self.conv_bn_relu(x3, "block3", 1, 3, 1, 1, 64)
        x3 = self.conv_bn_relu(x3, "block3", 2, 1, 1, 0, 64)

        # block4 -> x4 at 1/16
        x4 = self.conv_bn_relu(x3, "block4", 0, 3, 2, 1, 64)
        x4 = self.conv_bn_relu(x4, "block4", 1, 3, 1, 1, 64)
        x4 = self.conv_bn_relu(x4, "block4", 2, 3, 1, 1, 64)

        # block5 -> x5 at 1/32
        x5 = self.conv_bn_relu(x4, "block5", 0, 3, 2, 1, 128)
        x5 = self.conv_bn_relu(x5, "block5", 1, 3, 1, 1, 128)
        x5 = self.conv_bn_relu(x5, "block5", 2, 3, 1, 1, 128)
        x5 = self.conv_bn_relu(x5, "block5", 3, 1, 1, 0, 64)

        # pyramid fusion: x3 + up(x4) + up(x5)
        x4r = F.interpolate(x4, size=(x3.shape[2], x3.shape[3]),
                            mode="bilinear", align_corners=False)
        x5r = F.interpolate(x5, size=(x3.shape[2], x3.shape[3]),
                            mode="bilinear", align_corners=False)
        fused = x3 + x4r + x5r

        # block_fusion -> feats
        fx = self.conv_bn_relu(fused, "block_fusion", 0, 3, 1, 1, 64)
        fx = self.conv_bn_relu(fx, "block_fusion", 1, 3, 1, 1, 64)
        feats = self.conv1(fx, self.t["block_fusion.2.weight"],
                           self.t["block_fusion.2.bias"])

        # heatmap head
        hh = self.conv_bn_relu(feats, "heatmap_head", 0, 1, 1, 0, 64)
        hh = self.conv_bn_relu(hh, "heatmap_head", 1, 1, 1, 0, 64)
        heatmap = self.conv1(hh, self.t["heatmap_head.2.weight"],
                             self.t["heatmap_head.2.bias"])
        heatmap = torch.sigmoid(heatmap)

        # keypoint head on the 8x8-unfolded normalized input
        u = x.unfold(2, 8, 8).unfold(3, 8, 8)  # [1,1,H/8,W/8,8,8]
        u = u.permute(0, 1, 4, 5, 2, 3).reshape(1, 64, hc // 8, wc // 8)
        u = self.conv_bn_relu(u, "keypoint_head", 0, 1, 1, 0, 64)
        u = self.conv_bn_relu(u, "keypoint_head", 1, 1, 1, 0, 64)
        u = self.conv_bn_relu(u, "keypoint_head", 2, 1, 1, 0, 64)
        keypoints = self.conv1(u, self.t["keypoint_head.3.weight"],
                               self.t["keypoint_head.3.bias"])

        return feats, keypoints, heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, default="weights/xfeat.bin",
                        help="Binary weight file from export_xfeat_weights.py")
    parser.add_argument("--output", type=str, default="weights/xfeat.onnx")
    args = parser.parse_args()

    t = read_bin(args.bin)
    net = XFeatOnnx(t).eval()

    dummy = torch.rand(1, 1, 480, 640)  # non-multiple-of-32 to test resize
    with torch.no_grad():
        feats, keypoints, heatmap = net(dummy)
        print("shapes:", tuple(feats.shape), tuple(keypoints.shape),
              tuple(heatmap.shape))

    torch.onnx.export(
        net, dummy, args.output, verbose=False, do_constant_folding=True,
        input_names=["images"],
        output_names=["feats", "keypoints", "heatmap"],
        opset_version=18,
        dynamic_axes={
            "images": {2: "height", 3: "width"},
            "feats": {2: "height", 3: "width"},
            "keypoints": {2: "height", 3: "width"},
            "heatmap": {2: "height", 3: "width"},
        },
    )

    # Embed any external weights so the result is a single self-contained file
    # (matches the SuperPoint model and avoids .data sidecar lookup issues).
    import os
    import onnx
    embedded = args.output + ".embedded.onnx"
    m = onnx.load(args.output)
    onnx.save(m, embedded, save_as_external_data=False)
    os.replace(embedded, args.output)
    sidecar = args.output + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print("ONNX model saved to:", args.output)


if __name__ == "__main__":
    main()
