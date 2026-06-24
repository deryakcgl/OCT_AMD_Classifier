# run one oct image through my onnx model
# i exported octnet to onnx in export_onnx.py so the app does not need full pytorch at runtime
# torch stays in requirements because the training scripts still use it

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONNX_PATH = os.path.join(ROOT, "octnet.onnx")

# same 4 classes as training folder names
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]


@dataclass
class Prediction:
    class_label: str
    confidence: float
    probabilities: dict[str, float]


class OCTClassifier:
    def __init__(self, onnx_path: str = DEFAULT_ONNX_PATH):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        # InferenceSession loads the model once — the app creates one instance at startup
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image_path: str) -> np.ndarray:
        # must match training transforms or the results will be wrong
        # grayscale, 224x224, values 0–1 — same as train.py and onnx_infer.py
        img = Image.open(image_path).convert("L")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr[np.newaxis, np.newaxis, :, :]  # shape [1, 1, 224, 224]
        return arr

    def predict(self, image_path: str) -> Prediction:
        x = self._preprocess(image_path)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return Prediction(
            class_label=CLASSES[idx],
            confidence=float(probs[idx]),
            probabilities={CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax — i saw this approach in my notes
    e = np.exp(x - np.max(x))
    return e / e.sum()
