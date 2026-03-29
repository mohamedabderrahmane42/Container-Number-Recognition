import os
from pathlib import Path

# ── Project Root ──────────────────────────────
# src/config.py -> parent is src -> parent is root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"
RAW_PLATES  = DATA_DIR / "License_Plates" / "License_Plates_of_Algeria_Dataset-master"
MATRICULES  = DATA_DIR / "Matricules"
YOLO_OUTPUT = DATA_DIR / "yolo_dataset"

# ── Container Dataset Paths ───────────────────────
CONTAINER_DATA_DIR = PROJECT_ROOT / "Container Code Detection.v1i.yolov8"
CONTAINER_YAML     = CONTAINER_DATA_DIR / "data.yaml"
CONTAINER_TEST_DATA = CONTAINER_DATA_DIR / "test" / "images"

# ── Model Weights ─────────────────────────────
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OCR_MODEL   = WEIGHTS_DIR / "ocr_model.pt"
YOLO_BASE   = WEIGHTS_DIR / "yolov8n.pt"

# ── Container Model Weights ───────────────────────
CONTAINER_BEST      = WEIGHTS_DIR / "container_best.pt"
CONTAINER_OCR_MODEL = WEIGHTS_DIR / "container_ocr_best.pt"

# The trained YOLOv8 model - prioritizes the newest run from the Roboflow dataset
YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "detector_algerian500_v150" / "weights" / "best.pt"

if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "detector_algerian500" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det3" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det2" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det" / "weights" / "best.pt"

# ── Outputs ───────────────────────────────────
OUTPUT_DIR  = PROJECT_ROOT / "output"
CROPS_DIR   = OUTPUT_DIR / "cropped_plates"
TEST_RES    = OUTPUT_DIR / "test_results"
CSV_PATH    = OUTPUT_DIR / "results.csv"

# ── Container Output Paths ───────────────────────
CONTAINER_OUTPUT_DIR = OUTPUT_DIR / "container_results"
CONTAINER_CROPS_DIR  = CONTAINER_OUTPUT_DIR / "crops"
CONTAINER_CSV_PATH   = CONTAINER_OUTPUT_DIR / "results.csv"

# ── OCR Params ────────────────────────────────
CHARSET_NUMERIC = '0123456789'
CHARSET_ALPHANUMERIC = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'

# Default values (can be overridden by mode)
CHARSET   = CHARSET_NUMERIC
BLANK_IDX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1
IMG_H, IMG_W = 32, 128

# Weight Paths
OCR_MODEL   = WEIGHTS_DIR / "ocr_model.pt"
CONTAINER_OCR_CRNN = WEIGHTS_DIR / "container_ocr_crnn.pt"
CONTAINER_EXPERT_OCR = WEIGHTS_DIR / "container_expert_crnn.pt"
