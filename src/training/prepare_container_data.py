import os
import random
from pathlib import Path
from src.config import CONTAINER_DATA_DIR, DATA_DIR, PROJECT_ROOT


def prepare_container_dataset(train_ratio: float = 0.8):
    """Prepare the container dataset for YOLO training.

    This script expects the raw images to be placed in
    ``CONTAINER_DATA_DIR / "images"`` and the corresponding YOLO label files
    (``.txt``) in ``CONTAINER_DATA_DIR / "labels"``. It will:

    1. Verify the directories exist.
    2. Create ``train`` and ``val`` splits of the image filenames.
    3. Write ``train.txt`` and ``val.txt`` files listing the relative image paths.
    4. Generate a minimal ``container.yaml`` dataset description for Ultralytics.
    """
    images_dir = CONTAINER_DATA_DIR / "images"
    labels_dir = CONTAINER_DATA_DIR / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not image_files:
        raise RuntimeError("No image files found in the container dataset.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Write split files (relative to the dataset root)
    split_dir = CONTAINER_DATA_DIR
    with open(split_dir / "train.txt", "w") as f:
        for p in train_files:
            f.write(str(p.relative_to(CONTAINER_DATA_DIR)) + "\n")
    with open(split_dir / "val.txt", "w") as f:
        for p in val_files:
            f.write(str(p.relative_to(CONTAINER_DATA_DIR)) + "\n")

    # Create dataset yaml for Ultralytics
    yaml_path = CONTAINER_DATA_DIR / "container.yaml"
    yaml_content = f"""
path: {CONTAINER_DATA_DIR}
train: train.txt
val: val.txt
names:
  0: container_code
"""
    yaml_path.write_text(yaml_content.strip())
    print(f"Dataset preparation complete. YAML saved to {yaml_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare container dataset for YOLO training.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of data for training set")
    args = parser.parse_args()
    prepare_container_dataset(train_ratio=args.train_ratio)
