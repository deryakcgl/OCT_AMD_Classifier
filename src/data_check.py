# quick sanity check before training
# makes sure data folder structure is ok and batch shapes look right

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

DATA = os.path.expanduser("~/oct-amd-classifier/data")

# oct images are grayscale — resize to 224 and convert to tensor [0,1]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ImageFolder uses folder name as class label automatically
train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), transform=transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA, "test"),  transform=transform)

print("Classes:", train_ds.classes)  # expect ['CNV','DME','DRUSEN','NORMAL']
print("Train image count:", len(train_ds))
print("Test image count:", len(test_ds))

# load one batch and check dimensions
loader = DataLoader(train_ds, batch_size=16, shuffle=True)
images, labels = next(iter(loader))
print("Batch shape:", images.shape)   # expect [16, 1, 224, 224]
print("Labels:", labels.tolist())
print("Pixel range:", images.min().item(), "-", images.max().item())  # expect ~0.0 - 1.0
