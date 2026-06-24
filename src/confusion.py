# confusion matrix on test set — see which classes get mixed up
# saves plot to confusion_matrix.png in project root

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import OCTNet

DATA = os.path.expanduser("~/oct-amd-classifier/data")
FP32_PATH = os.path.expanduser("~/oct-amd-classifier/octnet_fp32.pth")
OUT_PNG = os.path.expanduser("~/oct-amd-classifier/confusion_matrix.png")

model = OCTNet()
model.load_state_dict(torch.load(FP32_PATH))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor(),
])
test_ds = datasets.ImageFolder(os.path.join(DATA,"test"), transform=transform)
loader = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes  # ['CNV','DME','DRUSEN','NORMAL']

# rows = true label, columns = predicted
cm = np.zeros((4,4), dtype=int)
with torch.no_grad():
    for imgs, labels in loader:
        preds = model(imgs).argmax(1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t][p] += 1

print("Confusion Matrix (row=true, col=predicted)")
print("Classes:", classes)
print(cm)
print()
# recall per class — how many of each true class we got right
for i, c in enumerate(classes):
    total = cm[i].sum()
    correct = cm[i][i]
    print(f"{c}: {correct}/{total} correct ({100*correct/total:.1f}%)")

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('OCT AMD Classifier - Confusion Matrix')
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i][j], ha='center', va='center',
                color='white' if cm[i][j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120)
print(f"\nPlot saved: {OUT_PNG}")
