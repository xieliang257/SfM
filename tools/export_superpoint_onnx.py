# -*- coding: utf-8 -*-
"""
Export SuperPoint model to ONNX format.

Usage:
    python3 tools/export_superpoint_onnx.py --weights weights/superpoint_v1.pth \
        --output weights/superpoint.onnx
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointNet(nn.Module):
    """SuperPoint network architecture."""

    def __init__(self):
        super().__init__()
        # Shared encoder (VGG-style)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Detector head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1)

        # Descriptor head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (1, 1, H, W) grayscale image tensor
        Returns:
            semi: (1, 65, H/8, W/8) semi-dense keypoint heatmap
            desc: (1, 256, H/8, W/8) dense descriptor map
        """
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head
        semi = self.convPb(self.relu(self.convPa(x)))

        # Descriptor head
        desc = self.convDb(self.relu(self.convDa(x)))
        desc = F.normalize(desc, p=2, dim=1)

        return semi, desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/superpoint_v1.pth")
    parser.add_argument("--output", type=str, default="weights/superpoint.onnx")
    args = parser.parse_args()

    # Load model
    model = SuperPointNet()
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    # Export
    dummy = torch.randn(1, 1, 240, 320)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"],
        output_names=["semi", "desc"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "semi": {0: "batch", 2: "height", 3: "width"},
            "desc": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=11,
    )
    print(f"exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
