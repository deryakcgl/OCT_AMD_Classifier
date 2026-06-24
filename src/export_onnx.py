# export trained pytorch model to onnx format
# why onnx? so inference.py and the desktop app can run without loading full pytorch
# onnxruntime is lighter for prediction only

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from model import OCTNet

FP32_PATH = os.path.expanduser("~/oct-amd-classifier/octnet_fp32.pth")
ONNX_PATH = os.path.expanduser("~/oct-amd-classifier/octnet.onnx")

model = OCTNet()
model.load_state_dict(torch.load(FP32_PATH))
model.eval()

# dummy input shape must match real preprocessing: batch 1, 1 channel, 224x224
dummy = torch.randn(1, 1, 224, 224)

torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # batch size can vary
    opset_version=13,
)
print(f"ONNX exported: {ONNX_PATH}")
print(f"ONNX size: {os.path.getsize(ONNX_PATH)/1e6:.2f} MB")
