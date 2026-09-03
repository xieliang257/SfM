# -*- coding: utf-8 -*-
"""
Convert the XFeat (CVPR 2024) PyTorch checkpoint into a compact binary weight
file that the C++ XFeat engine can load.

Only the convolutional backbone weights are exported (the sparse-matching
post-processing is implemented in C++). BatchNorm (affine=False) weights are
pre-folded into per-channel scale/shift vectors.

Binary layout (little-endian):
    [int32 n_tensors]
    for each tensor:
        [int32 name_len][name bytes]
        [int32 ndim]
        [int32 dims[ndim]]
        [float32 data[prod(dims)]]   (row-major, CxHxW for conv weights)

Usage:
    python3 tools/export_xfeat_weights.py --weights weights/xfeat.pt \
        --output weights/xfeat.bin
"""

import argparse
import struct

import torch


def write_tensor(f, name, tensor):
    name_b = name.encode("ascii")
    f.write(struct.pack("<i", len(name_b)))
    f.write(name_b)
    f.write(struct.pack("<i", tensor.dim()))
    for d in tensor.shape:
        f.write(struct.pack("<i", d))
    f.write(tensor.contiguous().float().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/xfeat.pt")
    parser.add_argument("--output", type=str, default="weights/xfeat.bin")
    args = parser.parse_args()

    sd = torch.load(args.weights, map_location="cpu")

    keep = []
    for key in sorted(sd.keys()):
        if "fine_matcher" in key:
            continue
        if key.endswith("num_batches_tracked"):
            continue
        keep.append(key)

    # BatchNorm (affine=False): y = scale * (x - mean) / sqrt(var + eps)
    # Store scale and shift instead of raw statistics.
    eps = 1e-5
    bn_scale = {}
    bn_shift = {}
    bn_keys = set()
    for key in list(keep):
        if key.endswith(".layer.1.running_mean"):
            bn_key = key[:-len("running_mean")]
            mean = sd[key]
            var = sd[bn_key + "running_var"]
            scale = (var + eps).rsqrt()
            bn_scale[bn_key] = scale
            bn_shift[bn_key] = -mean * scale
            bn_keys.add(bn_key)
            keep.remove(key)
            keep.remove(bn_key + "running_var")

    out_keys = []
    for key in keep:
        out_keys.append(key)
    for key in sorted(bn_scale.keys()):
        out_keys.append(key + "scale")
        out_keys.append(key + "shift")

    with open(args.output, "wb") as f:
        f.write(struct.pack("<i", len(out_keys)))
        for key in out_keys:
            if key in bn_scale:
                write_tensor(f, key, bn_scale[key])
            elif key in bn_shift:
                write_tensor(f, key, bn_shift[key])
            else:
                write_tensor(f, key, sd[key])

    total = sum(1 for _ in range(1))
    print("exported {} tensors to {}".format(len(out_keys), args.output))


if __name__ == "__main__":
    main()
