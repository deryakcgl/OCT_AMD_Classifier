# copy uploaded oct images into data/patients/ so paths stay valid
# if the user moves or deletes the original file the visit image should still load
# seed_data.py and app.py both use this

from __future__ import annotations

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_DIR = os.path.join(ROOT, "data", "patients")


def copy_visit_image(patient_id: int, visit_number: int, src_path: str) -> str:
    dest_dir = os.path.join(PATIENTS_DIR, f"patient_{patient_id}")
    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(src_path)[1] or ".jpeg"
    dest = os.path.join(dest_dir, f"visit_{visit_number}{ext}")
    shutil.copy2(src_path, dest)
    return dest
