# try int8 quantization on the fp32 model
# goal: smaller file + maybe faster on cpu
# i only quantized Linear layers (dynamic quant) — simplest approach i found
#
# note: octnet_int8.pth is experimental — the clinic app uses onnx, not this file
# i keep the script to compare size/speed but nothing loads int8 at runtime yet

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import OCTNet

torch.backends.quantized.engine = 'qnnpack'  # for cpu on mac

DATA = os.path.expanduser("~/oct-amd-classifier/data")
FP32_PATH = os.path.expanduser("~/oct-amd-classifier/octnet_fp32.pth")
INT8_PATH = os.path.expanduser("~/oct-amd-classifier/octnet_int8.pth")

# load fp32 weights first
model = OCTNet()
model.load_state_dict(torch.load(FP32_PATH))
model.eval()

# dynamic quantization — converts Linear weights to int8 at runtime
# conv layers stay fp32 in this setup
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

torch.save(quantized.state_dict(), INT8_PATH)

# --- measurements: size, accuracy, speed ---
fp32_mb = os.path.getsize(FP32_PATH)/1e6
int8_mb = os.path.getsize(INT8_PATH)/1e6
print(f"FP32 size: {fp32_mb:.2f} MB")
print(f"INT8 size: {int8_mb:.2f} MB")
print(f"Size reduction: {100*(1-int8_mb/fp32_mb):.1f}%")

transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor(),
])
test_ds = datasets.ImageFolder(os.path.join(DATA,"test"), transform=transform)

# accuracy after quantization — should be close to fp32
correct=total=0
with torch.no_grad():
    for img,lab in DataLoader(test_ds,batch_size=32):
        pred=quantized(img).argmax(1)
        correct+=(pred==lab).sum().item(); total+=lab.size(0)
print(f"INT8 accuracy: {100*correct/total:.1f}%")

# speed — warm up first few images then time 200
loader = DataLoader(test_ds, batch_size=1, shuffle=False)
with torch.no_grad():
    for i,(img,_) in enumerate(loader):
        quantized(img)
        if i>=5: break
times=[]
with torch.no_grad():
    for i,(img,_) in enumerate(loader):
        t0=time.time(); quantized(img); times.append(time.time()-t0)
        if i>=200: break
print(f"INT8 inference: {1000*sum(times)/len(times):.2f} ms/image")
print("--- QUANTIZED (INT8) ---")
