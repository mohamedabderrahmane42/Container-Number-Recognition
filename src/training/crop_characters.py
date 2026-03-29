import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from src.config import CONTAINER_BEST, PROJECT_ROOT

def main():
    """
    Auto-crop container blocks from main images to prepare for character-level labeling.
    Saves results to data/character_labeling/
    """
    # 1. Load models
    if not CONTAINER_BEST.exists():
        print(f"[ERROR] Block detector not found at {CONTAINER_BEST}. Train it first!")
        return
    
    model = YOLO(CONTAINER_BEST)
    
    # 2. Setup I/O
    input_dir = PROJECT_ROOT / "Container Code Detection.v1i.yolov8" / "train" / "images"
    if not input_dir.exists():
        print(f"[ERROR] Input images folder not found: {input_dir}")
        return
        
    output_dir = PROJECT_ROOT / "data" / "character_labeling"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Process
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(images)} images. Cropping codes...")
    
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        results = model(img, verbose=False)
        for i, result in enumerate(results):
            for box in result.boxes:
                if float(box.conf[0]) < 0.5: continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Add slight padding
                h, w = img.shape[:2]
                pad = 10
                crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                out_name = f"crop_{count:04d}_{img_path.name}"
                cv2.imwrite(str(output_dir / out_name), crop)
                count += 1
                
    print(f"Done! Saved {count} character-block crops to {output_dir}")
    print("Next Step: Upload these crops to Roboflow to label individual characters A-Z, 0-9.")

if __name__ == "__main__":
    main()
