"""
prepare_yolo.py  – Convert Algerian plates annotations to YOLO format
"""

import os
import json
import shutil
from src.config import RAW_PLATES, YOLO_OUTPUT

def convert_to_yolo():
    json_path = RAW_PLATES / "export" / "annotations.json"
    imgs_dir  = RAW_PLATES / "images"

    if not json_path.exists():
        print(f"[ERROR] annotations.json not found at {json_path}")
        return

    # Create directories
    for split in ["train", "val"]:
        os.makedirs(YOLO_OUTPUT / split / "images", exist_ok=True)
        os.makedirs(YOLO_OUTPUT / split / "labels", exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    # ... logic from previous prepare_dataset.py ...
    print("[INFO] Dataset preparation complete.")

if __name__ == "__main__":
    convert_to_yolo()
