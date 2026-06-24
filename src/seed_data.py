# sample patients so the app is not empty on first open
# copies images from data/train or data/test into data/patients/
# runs onnx on them and saves the results to sqlite
#
# python seed_data.py
# python seed_data.py --force   — delete db and start over

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from db import add_patient, add_visit, init_db, list_patients
from inference import OCTClassifier
from patient_storage import PATIENTS_DIR, copy_visit_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# example names and complaints with real oct files from the dataset
# patient 1 has a class change (drusen → dme) — useful for testing genai
SEED = [
    {
        "name": "Reyhan KO",
        "visits": [
            ("2024-01-15", "Blurred vision in right eye for 2 months", "DRUSEN", "DRUSEN-1083159-1.jpeg"),
            ("2024-06-20", "Vision has worsened further", "DME", "DME-1081406-1.jpeg"),
        ],
    },
    {
        "name": "Metin GU",
        "visits": [
            ("2024-02-10", "Central vision loss", "CNV", "CNV-1016042-1.jpeg"),
            ("2024-08-05", "Post-treatment follow-up", "CNV", "CNV-1016042-2.jpeg"),
        ],
    },
    {
        "name": "NECLA DA",
        "visits": [
            ("2024-03-01", "Routine eye exam", "NORMAL", "NORMAL-1017237-1.jpeg"),
            ("2024-09-12", "Annual check-up", "NORMAL", "NORMAL-1025847-1.jpeg"),
        ],
    },
]


def _find_image(class_folder: str, filename: str) -> str:
    # folder name in the dataset matches the class label
    for split in ("test", "train"):
        path = os.path.join(DATA, split, class_folder, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Image not found: {class_folder}/{filename}")



def seed(force: bool = False) -> None:
    init_db()
    if list_patients() and not force:
        print("Database already has patients. Re-seed with: python seed_data.py --force")
        return

    if force:
        # delete existing db and patient images when i want a clean start
        db_path = os.path.join(ROOT, "oct_clinic.db")
        if os.path.isfile(db_path):
            os.remove(db_path)
        if os.path.isdir(PATIENTS_DIR):
            shutil.rmtree(PATIENTS_DIR)
        init_db()

    clf = OCTClassifier()
    print("Creating sample data...")

    for spec in SEED:
        pid = add_patient(spec["name"])
        print(f"  + {spec['name']} (id={pid})")
        for i, (vdate, complaints, cls_folder, fname) in enumerate(spec["visits"], 1):
            src = _find_image(cls_folder, fname)
            dest = copy_visit_image(pid, i, src)
            pred = clf.predict(dest)  # save real model output, not just the folder label
            add_visit(
                patient_id=pid,
                visit_date=vdate,
                complaints=complaints,
                image_path=dest,
                class_label=pred.class_label,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
            )
            print(f"      visit {vdate}: {pred.class_label} ({pred.confidence * 100:.0f}%)")

    print(f"\nDone. DB: {os.path.join(ROOT, 'oct_clinic.db')}")


if __name__ == "__main__":
    seed(force="--force" in sys.argv)
