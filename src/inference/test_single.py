"""
test_single.py – Interactive script to test the model on a single image.
"""

import sys
import os
import cv2
import torch
import numpy as np

from src.config import OUTPUT_DIR, CONTAINER_BEST, CONTAINER_OUTPUT_DIR, CONTAINER_OCR_MODEL
from src.config import OUTPUT_DIR, CONTAINER_OUTPUT_DIR
from src.inference.pipeline import load_models, read_crnn, read_easyocr
from src.utils.formatters import format_container_code, format_algerian

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.test_single path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = os.getenv('DETECT_MODE', 'plate')
    print(f"--- Testing {mode.upper()} mode on {device} ---")

    # Load models using unified helper
    try:
        yolo_model, ocr_engine, charset = load_models(device, mode=mode)
    except Exception as e:
        print(f"\n[ERROR] Models could not be loaded: {e}")
        return

    # Process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h_img, w_img = img.shape[:2]
    results = yolo_model(img, verbose=False)
    
    found = False
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.4: continue
                
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop with some padding
            pad = 15
            crop = img[max(0, y1-pad):min(h_img, y2+pad), max(0, x1-pad):min(w_img, x2+pad)]
            
            # Recognition Stage
            if mode == 'container':
                ocr_type = os.getenv('OCR_TYPE', 'precise')
                if ocr_type == 'expert':
                    raw_text = read_crnn(crop, ocr_engine, charset, device)
                else:
                    raw_text = read_easyocr(crop, ocr_engine)
                
                display_text = format_container_code(raw_text)
                color = (255, 0, 0) # Blue
            else:
                raw_text = read_crnn(crop, ocr_engine, charset, device)
                display_text = format_algerian(raw_text)
                color = (0, 200, 0) # Green
            
            # Visualize
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            label = f" {display_text} "
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img, (x1, y1-th-20), (x1+tw+8, y1), color, -1)
            text_color = (255, 255, 255) if mode == 'container' else (0, 0, 0)
            cv2.putText(img, label, (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
            
            print(f"Detected: {display_text} (raw: {raw_text}, conf={conf:.2f})")

    if not found:
        print(f"No {mode} detected.")
        return

    # Save output visualization
    if w_img > 1400:
        scale = 1400 / w_img
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        
    out_dir = CONTAINER_OUTPUT_DIR if mode == 'container' else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"test_{os.path.basename(image_path)}"
    
    cv2.imwrite(str(out_path), img)
    print(f"\nVisualization saved to: {out_path}")

if __name__ == '__main__':
    main()
