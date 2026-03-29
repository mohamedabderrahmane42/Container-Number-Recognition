import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from src.models.crnn import CRNN
from src.utils.formatters import decode_crnn
from src.config import CONTAINER_EXPERT_OCR, CHARSET_ALPHANUMERIC, IMG_H, IMG_W

def test_expert():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Expert CNN on {device}...")
    
    # Load Expert Model
    num_classes = len(CHARSET_ALPHANUMERIC) + 1
    model = CRNN(num_classes).to(device)
    
    if not CONTAINER_EXPERT_OCR.exists():
        print(f"Weights not found: {CONTAINER_EXPERT_OCR}")
        return
        
    ckpt = torch.load(CONTAINER_EXPERT_OCR, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Process image
    img_path = 'Container Code Detection.v1i.yolov8/test/images/1-122720001-OCR-AS-B01_jpg.rf.a3a400a6fe971f634426a3ed5661fb9d.jpg'
    img = cv2.imread(img_path)
    # Note: We need a crop, but let's just resize the whole img to test logic
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    
    tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(tensor)
        raw = decode_crnn(preds[:, 0, :].cpu().numpy(), charset=CHARSET_ALPHANUMERIC)
        
    print(f"Expert Prediction: {raw}")

if __name__ == "__main__":
    test_expert()
