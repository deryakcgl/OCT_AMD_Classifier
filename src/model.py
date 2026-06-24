# oct cnn — one place for the architecture so train/export/quantize stay in sync
# i copied this class in four files before and that was asking for silent mismatches
#
# note to self: most parameters live in Linear(64*28*28, 128) — roughly 6.4M of them
# the conv blocks are small. global average pooling instead of flatten could shrink
# the model a lot and maybe help overfitting — something to try later

from __future__ import annotations

import torch.nn as nn


class OCTNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
