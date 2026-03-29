import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from src.config import (
    RAW_PLATES, DATA_DIR, 
    OCR_MODEL, CONTAINER_OCR_CRNN, CONTAINER_EXPERT_OCR,
    CONTAINER_CSV_PATH,
    CHARSET_NUMERIC, CHARSET_ALPHANUMERIC,
    IMG_H, IMG_W
)
from src.models.crnn import CRNN
from src.data.dataset_ocr import PlateDataset, crnn_collate_fn
from src.utils.formatters import decode_crnn

BATCH_SIZE = 32
NUM_EPOCHS = 500
LR         = 5e-4

def evaluate(model, loader, device, charset):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, _, _, raw_labels in loader:
            images   = images.to(device)
            preds    = model(images)
            preds_np = preds.cpu().numpy()
            T, B, C  = preds_np.shape
            for b in range(B):
                pred_str = decode_crnn(preds_np[:, b, :], charset=charset)
                true_str = raw_labels[b]
                if pred_str.strip() == true_str.strip():
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0

def train(mode='plate'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training OCR mode: {mode.upper()} on {device} ---")

    # Select Mode Parameters
    if mode == 'container':
        charset = CHARSET_ALPHANUMERIC
        # Prioritize Expert weights if they exist, else use filename weights or scratch
        weight_path = CONTAINER_EXPERT_OCR
        csv_path = CONTAINER_CSV_PATH if CONTAINER_CSV_PATH.exists() else None
        
        if csv_path:
            print("  [EXPERT MODE] Training with Silver Labels from CSV...")
        else:
            print("  [LEGACY MODE] No CSV found. Training on Filenames...")
            
        data_root = os.path.join(DATA_DIR, 'character_labeling')
        full_ds = PlateDataset(data_root, augment=True, mode='container', charset=charset, csv_path=csv_path)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    else:
        charset = CHARSET_NUMERIC
        weight_path = OCR_MODEL
        train_dir = os.path.join(RAW_PLATES, 'train')
        val_dir   = os.path.join(RAW_PLATES, 'validation')
        train_ds = PlateDataset(train_dir, augment=True, mode='plate', charset=charset)
        val_ds   = PlateDataset(val_dir, augment=False, mode='plate', charset=charset)

    num_classes = len(charset) + 1
    blank_idx = len(charset)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=crnn_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=crnn_collate_fn, num_workers=0)

    model     = CRNN(num_classes).to(device)
    ctc_loss  = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)

    if weight_path.exists():
        print(f"Loading existing weights from {weight_path}...")
        ckpt = torch.load(weight_path, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except:
            print("Weight mismatch (likely charset size change). Starting from scratch.")

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS}")
        for images, targets, target_lengths, _ in pbar:
            images  = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            T, B, C = preds.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = ctc_loss(preds, targets, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device, charset)
        scheduler.step(1 - val_acc)

        print(f"  loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes' : num_classes,
                'charset'     : charset,
                'blank_idx'   : blank_idx,
                'img_height'  : IMG_H,
                'img_width'   : IMG_W,
            }, weight_path)
            print(f"  >> Model saved to {weight_path.name} (acc={val_acc:.3f})")

    print(f"\nDone! Best val accuracy = {best_acc:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='plate', choices=['plate', 'container'])
    args = parser.parse_args()
    train(mode=args.mode)
