"""
pipeline.py  – Batch inference pipeline for Algerian License Plates
"""

import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path

import easyocr
from src.config import (
    MATRICULES,
    OUTPUT_DIR,
    CROPS_DIR,
    YOLO_BEST,
    OCR_MODEL,
    NUM_CLASSES,
    IMG_W,
    IMG_H,
    CONTAINER_BEST,
    CONTAINER_OUTPUT_DIR,
    CONTAINER_CROPS_DIR,
    CONTAINER_CSV_PATH,
    CONTAINER_OCR_CRNN,  # Use CRNN path
    CONTAINER_EXPERT_OCR,
    CONTAINER_TEST_DATA,
    CHARSET_NUMERIC,
    CHARSET_ALPHANUMERIC
)
from src.models.crnn import CRNN
from src.utils.formatters import decode_crnn, format_algerian, format_container_code

# Confidence thresholds
CONF_THRESHOLD = 0.50
BBOX_PADDING = 12

def load_ocr_model(weight_path, device, num_classes) -> nn.Module:
    """Generic CRNN loader."""
    model = CRNN(num_classes).to(device)
    if weight_path.exists():
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def load_models(device, mode='plate') -> Tuple[object, object, str]:
    """Load detector and corresponding OCR engine (Fast CNN or Precise EasyOCR)."""
    from ultralytics import YOLO
    
    ocr_type = os.getenv('OCR_TYPE', 'precise') # 'precise' (EasyOCR) or 'expert' (Custom CNN)

    if mode == 'container':
        print(f"Loading Container Block Detector: {CONTAINER_BEST}")
        yolo_model = YOLO(CONTAINER_BEST)
        
        if ocr_type == 'expert':
            print(f"Loading EXPERT CNN OCR: {CONTAINER_EXPERT_OCR}")
            charset = CHARSET_ALPHANUMERIC
            num_classes = len(charset) + 1
            ocr_engine = load_ocr_model(CONTAINER_EXPERT_OCR, device, num_classes)
        else:
            print("Initializing EasyOCR Reader (English)...")
            ocr_engine = easyocr.Reader(['en'], gpu=(device=='cuda'))
            charset = None 
    else:
        print(f"Loading License Plate Detector: {YOLO_BEST}")
        yolo_model = YOLO(YOLO_BEST)
        charset = CHARSET_NUMERIC
        num_classes = len(charset) + 1
        ocr_engine = load_ocr_model(OCR_MODEL, device, num_classes)

    return yolo_model, ocr_engine, charset

def read_crnn(img_crop: np.ndarray, model: nn.Module, charset: str, device: str) -> str:
    """Generic sequence reader using CRNN."""
    if len(img_crop.shape) == 3:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_crop = clahe.apply(img_crop)
    img_crop = cv2.resize(img_crop, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    
    img_tensor = torch.FloatTensor(img_crop).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds    = model(img_tensor)
        preds_np = preds[:, 0, :].cpu().numpy()
        
    return decode_crnn(preds_np, charset=charset)


def read_easyocr(img_crop: np.ndarray, reader: easyocr.Reader) -> str:
    """Read alphanumeric code using EasyOCR. Native color handling for robust boxed text."""
    if img_crop.size == 0: return ""
    
    # Sometimes slightly larger resolution helps deep-learning OCRs see the characters inside boxes
    h, w = img_crop.shape[:2]
    if h < 64:
        scale = 64 / h
        img_crop = cv2.resize(img_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # EasyOCR on the block
    results = reader.readtext(img_crop, detail=0)
    return " ".join(results).upper()


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input images directory')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = os.getenv('DETECT_MODE', 'plate')
    print(f"--- Running {mode.upper()} Pipeline (Precision EasyOCR) ---")
    
    # 15px is the "Sweet Spot" for ISO codes – avoids capturing the line below
    padding = 15 if mode == 'container' else 12

    try:
        yolo_model, ocr_engine, charset = load_models(device, mode=mode)
    except Exception as e:
        print(f"[ERROR] Loading models failed: {e}")
        return

    # Prepare I/O
    if mode == 'container':
        input_dir = Path(args.input) if args.input else CONTAINER_TEST_DATA
        os.makedirs(CONTAINER_OUTPUT_DIR, exist_ok=True)
        os.makedirs(CONTAINER_CROPS_DIR, exist_ok=True)
    else:
        input_dir = Path(args.input) if args.input else MATRICULES
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CROPS_DIR, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Get valid images
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    print(f"Processing {len(files)} images...\n")

    results_data = []
    processed = 0
    detected = 0

    for fname in files:
        img = cv2.imread(str(input_dir / fname))
        if img is None: continue
            
        processed += 1
        h_img, w_img = img.shape[:2]
        results = yolo_model(img, verbose=False)

        for result in results:
            for box in result.boxes:
                det_conf = float(box.conf[0])
                if det_conf < CONF_THRESHOLD: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Dynamic Padding
                x1p, y1p = max(0, x1-padding), max(0, y1-padding)
                x2p, y2p = min(w_img, x2+padding), min(h_img, y2+padding)

                crop = img[y1p:y2p, x1p:x2p]
                if crop.size == 0: continue
                detected += 1
                
                # Recognition Stage
                if mode == 'container':
                    ocr_type = os.getenv('OCR_TYPE', 'precise')
                    if ocr_type == 'expert':
                        raw_text = read_crnn(crop, ocr_engine, charset, device)
                    else:
                        raw_text = read_easyocr(crop, ocr_engine)
                        
                    formatted_text = format_container_code(raw_text)
                    save_prefix = "expert" if ocr_type == 'expert' else "easy"
                    save_path = CONTAINER_CROPS_DIR / f"{save_prefix}_{formatted_text}_{fname}"
                else:
                    raw_text = read_crnn(crop, ocr_engine, charset, device)
                    formatted_text = format_algerian(raw_text)
                    save_path = CROPS_DIR / f"plt_{formatted_text}_{fname}"
                
                results_data.append({
                    'image': fname,
                    'result': formatted_text,
                    'raw': raw_text,
                    'confidence': f"{det_conf:.2f}"
                })
                cv2.imwrite(str(save_path), crop)

        if processed % 10 == 0:
            print(f"  Progress: {processed}/{len(files)} images, {detected} detections")

    # Save to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = CONTAINER_CSV_PATH if mode == 'container' else (OUTPUT_DIR / "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Saved {len(results_data)} results to {csv_path}")
    else:
        print("\n[INFO] No detections found.")

if __name__ == '__main__':
    main()
