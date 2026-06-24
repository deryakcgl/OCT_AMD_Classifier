# check onnx model on the full test set
# compares accuracy and speed against pytorch baseline
# run after export_onnx.py

import onnxruntime as ort
import numpy as np
import torch, os, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ONNX_PATH = os.path.expanduser("~/oct-amd-classifier/octnet.onnx")
DATA = os.path.expanduser("~/oct-amd-classifier/data")

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor(),
])
test_ds = datasets.ImageFolder(os.path.join(DATA,"test"), transform=transform)
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# load onnx with onnxruntime — no pytorch forward pass needed
sess = ort.InferenceSession(ONNX_PATH)
input_name = sess.get_inputs()[0].name

correct = total = 0
times = []
for img, label in loader:
    x = img.numpy()  # shape [1, 1, 224, 224]
    t0 = time.time()
    out = sess.run(None, {input_name: x})[0]
    times.append(time.time()-t0)
    pred = int(np.argmax(out, axis=1)[0])
    correct += (pred == label.item())
    total += 1
    if total >= 968: break  # full test set size

print(f"ONNX Runtime accuracy: {100*correct/total:.1f}%")
print(f"ONNX Runtime inference: {1000*sum(times)/len(times):.2f} ms/image")
print("--- ONNX RUNTIME ---")
