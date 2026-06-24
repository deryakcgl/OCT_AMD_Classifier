# baseline numbers for fp32 pytorch model before quantization / onnx
# i run this to have something to compare against later

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import OCTNet

DATA = os.path.expanduser("~/oct-amd-classifier/data")
MODEL_PATH = os.path.expanduser("~/oct-amd-classifier/octnet_fp32.pth")

model = OCTNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 1) file size
size_mb = os.path.getsize(MODEL_PATH) / 1e6
print(f"Model size: {size_mb:.2f} MB")

# 2) inference speed on test images
transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor(),
])
test_ds = datasets.ImageFolder(os.path.join(DATA,"test"), transform=transform)
loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# warm-up — first runs are slower, skip them
with torch.no_grad():
    for i,(img,_) in enumerate(loader):
        model(img)
        if i>=5: break

times=[]
with torch.no_grad():
    for i,(img,_) in enumerate(loader):
        t0=time.time(); model(img); times.append(time.time()-t0)
        if i>=200: break
avg_ms = 1000*sum(times)/len(times)
print(f"Average inference: {avg_ms:.2f} ms/image ({len(times)} samples)")

# 3) accuracy check
correct=total=0
with torch.no_grad():
    for img,lab in DataLoader(test_ds,batch_size=32):
        pred=model(img).argmax(1)
        correct+=(pred==lab).sum().item(); total+=lab.size(0)
print(f"Test accuracy: {100*correct/total:.1f}%")
print("--- BASELINE (FP32) ---")
