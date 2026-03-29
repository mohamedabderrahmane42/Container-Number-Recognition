import os
from pathlib import Path
from ultralytics import YOLO
from src.config import CONTAINER_DATA_DIR, CONTAINER_BEST, WEIGHTS_DIR, YOLO_BASE, CONTAINER_YAML


def train_container_model(epochs: int = 50, batch: int = 16, img_size: int = 640, project_name: str = "container"):
    """Train a YOLO model for container code detection.

    Args:
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        img_size (int): Image size for training (YOLO expects square).
        project_name (str): Subdirectory under `runs/detect` where results are stored.
    """
    # Ensure dataset exists
    if not CONTAINER_DATA_DIR.exists():
        raise FileNotFoundError(f"Container dataset directory not found: {CONTAINER_DATA_DIR}")

    # YOLO expects a YAML file describing the dataset.
    yaml_path = CONTAINER_YAML
    if not yaml_path.exists():
        # Check for alternative 'container.yaml' or fallback
        yaml_path = CONTAINER_DATA_DIR / "container.yaml"
        
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {CONTAINER_YAML} or {yaml_path}")
        
    print(f"Using dataset yaml at: {yaml_path}")

    # Initialize model – start from base checkpoint
    model = YOLO(YOLO_BASE)
    # Train
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=str(Path.cwd() / "runs" / "detect"),
        name=project_name,
        exist_ok=True,
    )
    # After training, the best weights are saved under runs/detect/<project_name>/weights/best.pt
    best_weights = Path.cwd() / "runs" / "detect" / project_name / "weights" / "best.pt"
    if best_weights.exists():
        # Copy to our designated container best path
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        destination = CONTAINER_BEST
        destination.parent.mkdir(parents=True, exist_ok=True)
        best_weights.replace(destination)
        print(f"Best container model saved to {destination}")
    else:
        print("Training completed but best weights not found.")


if __name__ == "__main__":
    # Simple CLI interface
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO model for container code detection.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="container", help="Project name for runs folder")
    args = parser.parse_args()
    train_container_model(epochs=args.epochs, batch=args.batch, img_size=args.imgsz, project_name=args.name)
