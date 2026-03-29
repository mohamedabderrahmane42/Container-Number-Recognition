import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CHARSET, IMG_H, IMG_W

# Try importing albumentations for data augmentation
try:
    import albumentations as A
    A_AVAILABLE = True
except ImportError:
    A_AVAILABLE = False


def get_train_transforms():
    """Robust data augmentations for OCR training."""
    if not A_AVAILABLE:
        return None
    return A.Compose([
        # Random minor rotations (very common in YOLO crops)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.6, border_mode=cv2.BORDER_REPLICATE),
        # Lighting variations (contrast, brightness)
        A.RandomBrightnessContrast(p=0.5),
        # Noise / blur
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.4),
        # Simulating perspective distortion from angled viewing
        A.Perspective(scale=(0.02, 0.08), p=0.3, fit_output=True),
    ])


def parse_label(filename: str) -> str:
    """Extract the digit label from a license plate filename."""
    base = os.path.splitext(filename)[0]
    base = re.sub(r'_\d+$', '', base)
    base = re.sub(r'\s*\(\d+\)\s*$', '', base)
    digits = re.sub(r'[^0-9]', '', base)
    return digits


def parse_container_label(filename: str) -> str:
    """
    Extract the alphanumeric label from a container crop filename.
    Format: crop_0000_1-122700001-OCR-LF-C01_jpg.rf...
    Splits by '-OCR' and removes the prefix.
    """
    # Remove our prefix crop_xxxx_ (10 characters)
    if filename.startswith('crop_'):
        filename = filename[10:]
        
    # Split by -OCR or _jpg
    label = filename.split('-OCR')[0].split('_jpg')[0]
    return label


import pandas as pd

class PlateDataset(Dataset):
    """Refactored OCR dataset supporting plates, container filenames, and CSV silver labels."""
    def __init__(self, root_dir: str, augment: bool = False, mode: str = 'plate', 
                 charset: str = CHARSET, csv_path: str = None):
        self.samples = []
        self.augment = augment
        self.mode = mode
        self.charset = charset
        self.transforms = get_train_transforms() if augment else None
        
        extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        if not os.path.exists(root_dir):
             raise FileNotFoundError(f"Directory not found: {root_dir}")

        # --- OPTION 1: Use CSV for Ground Truth (Expert Mode) ---
        if csv_path and os.path.exists(csv_path):
            print(f"  Mapping dataset to Silver Labels from: {csv_path}")
            df = pd.read_csv(csv_path)
            # Create mapping: image_filename -> corrected_result
            # image column in CSV matched against the original image part of our crop filenames
            label_map = {}
            for _, row in df.iterrows():
                # Prioritize 'result' (Smart Corrected), fallback to 'raw' (EasyOCR Original)
                res = row['result'] if not pd.isna(row['result']) else row['raw']
                if pd.isna(res): continue
                
                # Clean up and ensure upper case
                label = re.sub(r'[^A-Z0-9]', '', str(res).upper())
                if len(label) >= 4:
                    label_map[row['image']] = label
            
            for fname in os.listdir(root_dir):
                if not fname.lower().endswith(extensions): continue
                
                # Extract the source image name from our crop filename
                if fname.startswith('crop_'):
                    source_name = fname[10:] # Remove 'crop_xxxx_'
                    if source_name in label_map:
                        self.samples.append((os.path.join(root_dir, fname), label_map[source_name]))
            
        # --- OPTION 2: Use Filenames for Ground Truth (Legacy/Initial) ---
        else:
            for fname in os.listdir(root_dir):
                if not fname.lower().endswith(extensions):
                    continue
                
                if mode == 'container':
                    label = parse_container_label(fname)
                    min_len = 3
                else:
                    label = parse_label(fname)
                    min_len = 8
                    
                if len(label) >= min_len:
                    self.samples.append((os.path.join(root_dir, fname), label))
                
        if not self.samples:
            raise RuntimeError(f"No valid {mode} samples found in: {root_dir} (CSV path: {csv_path})")
            
        print(f"  Loaded {len(self.samples)} Expert samples from {os.path.basename(root_dir)} ({mode} mode)")

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if self.transforms:
            img = self.transforms(image=img)["image"]
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img   = clahe.apply(img)
        img   = cv2.resize(img, (IMG_W, IMG_H))
        
        img   = img.astype(np.float32) / 255.0
        return img[np.newaxis, :]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        
        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            
        img_tensor = self.preprocess_image(img)
        
        # Use instance's charset
        label_ids = [self.charset.index(c) for c in label if c in self.charset]
        
        return torch.FloatTensor(img_tensor), label_ids, label


def crnn_collate_fn(batch):
    images, labels_list, raw_labels = zip(*batch)
    images         = torch.stack(images, 0)
    target_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.long)
    targets        = torch.tensor([c for lbl in labels_list for c in lbl], dtype=torch.long)
    return images, targets, target_lengths, raw_labels
