# build a smaller balanced dataset from the full OCT2017 archive
# full dataset was too heavy for my laptop experiments
# i keep 2000 per class for train, 242 per class for test

import os
import random
import shutil

# source = downloaded archive, destination = data/ used by train.py
# note: folder name has a trailing space in the original download path
SRC = os.path.expanduser("~/oct-amd-classifier/archive/OCT2017 ")
DST = os.path.expanduser("~/oct-amd-classifier/data")

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
PER_CLASS_TRAIN = 2000
random.seed(42)  # same subset if i run again

def build(split, per_class):
    for c in CLASSES:
        src_dir = os.path.join(SRC, split, c)
        dst_dir = os.path.join(DST, split, c)
        os.makedirs(dst_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        random.shuffle(files)
        chosen = files[:per_class]
        for f in chosen:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print(f"{split}/{c}: copied {len(chosen)}")

build("train", PER_CLASS_TRAIN)
build("test", 242)  # test set already balanced in the archive
print("Done. Subset is in data/")
