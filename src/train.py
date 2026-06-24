# train the oct classifier from scratch (or re-train)
# personal project — i am learning how the full pipeline fits together
# saves weights to octnet_fp32.pth when done
# deps: pip install -r requirements-train.txt

import os, time
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import OCTNet

DATA = os.path.expanduser("~/oct-amd-classifier/data")
DEVICE = torch.device("cpu")  # no gpu on my machine for now

# augmentation only on train — test must stay clean or accuracy means nothing
# for oct i kept it mild: horizontal flip + small brightness/contrast
# no vertical flip because up/down orientation matters in retinal scans i think
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), transform=train_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA, "test"),  transform=test_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

model = OCTNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# lr scheduler — halve learning rate every 4 epochs for finer tuning at the end
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# started with 5 epochs, increased to 12 after seeing loss still going down
EPOCHS = 12

def evaluate():
    # quick test-set accuracy between epochs
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            pred = model(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

for epoch in range(EPOCHS):
    model.train()
    t0 = time.time()
    running = 0.0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    scheduler.step()
    acc = evaluate()
    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} — loss {running/len(train_loader):.3f}, test acc {acc:.1f}%, lr {lr_now:.4f}, time {time.time()-t0:.0f}s")

torch.save(model.state_dict(), os.path.expanduser("~/oct-amd-classifier/octnet_fp32.pth"))
print("Model saved: octnet_fp32.pth")
