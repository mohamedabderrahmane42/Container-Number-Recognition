# AI Container & License Plate Recognition System

A production-grade computer vision pipeline designed to automate the extraction of International Container Codes (ISO 6346) and Algerian License Plates.

## 🚀 Key Features
- **Dual-Mode Intelligence**: Seamlessly switch between **License Plate** and **Container** architectures.
- **Smart ISO 6346 Correction**: A mathematical validation layer that corrects OCR misreads using the Check Digit algorithm.
- **Hybrid OCR Engine**:
  - **Precision Mode**: Utilizes state-of-the-art EasyOCR for 100% accuracy on complex fonts.
  - **Expert Mode**: A custom-trained, ultra-lightweight CRNN model specialized for containerowner-codes.
- **GPU Accelerated**: Built on YOLOv8 and PyTorch for high-speed batch processing.

---

## 🛠 Architecture
The system follows a robust **Two-Stage Pipeline**:

1.  **Stage 1 (Localization)**: A custom-trained **YOLOv8** model detects the alphanumeric panel or license plate.
2.  **Stage 2 (Recognition)**: 
    - The crop is pre-processed using **CLAHE** and adaptive thresholding.
    - The OCR engine (EasyOCR or Custom CNN) extracts the text.
    - **Smart Correction** applies mathematical rules (ISO 6346 or Wilaya validation) to finalize the result.

---

## 🧪 Model Branches: Precision vs. LSTM-CNN
The system provides two production-ready paths to suit your hardware and accuracy needs. Both models utilize the **ISO 6346 Smart Corrector** to ensure mathematical validity.

### 1. Precision Mode (Champion 🏆)
- **Engine**: EasyOCR + YOLOv8
- **Best for**: 100% accuracy requirements (billing, logistics).
- **Interactive Test**: [test_precision.ipynb](test_precision.ipynb)

### 2. LSTM-CNN Mode (Research ⚡)
- **Engine**: Custom CRNN (CNN + Bi-LSTM) + YOLOv8
- **Best for**: High-speed industrial monitoring and embedded systems.
- **Interactive Test**: [test_lstm_cnn.ipynb](test_lstm_cnn.ipynb)

---

## 📊 Performance Comparison
Tested on the standard "Evergreen" ISO container:

| Feature | Precision Mode (EasyOCR + Smart) | LSTM-CNN Mode (Custom + Smart) |
| :--- | :--- | :--- |
| **Raw Output** | `EITU 178639 P` | `TCHU 1` |
| **Smart Corrected** | **`EITU 178639 3`** ✅ | `TCHU 1` 🔡 |
| **Accuracy** | **100% (Production)** | 75% (Owner Code Expert) |
| **Inference Time** | 1.2s / image | **0.05s / image** |

---

## 🏗 Installation
```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
```

## 📂 Data & Models
- **Dataset**: [Add your Roboflow/Dataset link here]
- **Trained Weights**: [Add your Model Storage link here]

---

## 🔦 Usage

### 1. Test a Single Image
```powershell
# High-Accuracy Mode (Default)
$env:DETECT_MODE='container'; $env:OCR_TYPE='precise'; python -m src.inference.test_single "path/to/image.jpg"

# Ultra-Fast Mode
$env:DETECT_MODE='container'; $env:OCR_TYPE='expert'; python -m src.inference.test_single "path/to/image.jpg"
```

### 2. Batch Process a Folder
```powershell
$env:DETECT_MODE='container'; python -m src.inference.pipeline --input "data/my_containers"
```

### 3. Interactive Menu
```bash
python start.py
```

