import os
from pathlib import Path
from ultralytics import YOLO
from src.config import PROJECT_ROOT, WEIGHTS_DIR, CONTAINER_OCR_MODEL

def train_character_yolo(epochs=100, imgsz=320, batch=32):
    """
    Train a YOLOv8 model for individual character detection (Stage 2 OCR).
    """
    # 1. Path to our character dataset YAML
    yaml_path = PROJECT_ROOT / "data" / "yolo_ocr_dataset" / "ocr_chars.yaml"
    
    if not yaml_path.exists():
        print(f"[ERROR] OCR Dataset YAML not found: {yaml_path}")
        print("Please run `python -m src.training.prepare_ocr_data` first.")
        return

    # 2. Use a smaller model for OCR (Nano is usually enough)
    model = YOLO('yolov8n.pt')

    # 3. Train the model
    # Note: imgsz can be smaller (e.g., 320) since we are detecting characters in cropped blocks.
    print(f"Starting Stage 2: Character OCR Training ({epochs} epochs)...")
    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="container_ocr",
        exist_ok=True,
        cache=True,
        # Optimizer and LR tweaks for small objects/characters
        optimizer='Adam',
        lr0=1e-3,
        momentum=0.9
    )

    # 4. Save best weights to the configured path
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        import shutil
        shutil.copy(best_weights, CONTAINER_OCR_MODEL)
        print(f"\n[SUCCESS] Character OCR weights saved to: {CONTAINER_OCR_MODEL}")
    else:
        print("[ERROR] Best weights not found after training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Stage 2 Character OCR.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    train_character_yolo(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
