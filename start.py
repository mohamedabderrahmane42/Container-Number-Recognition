import os
import sys
from pathlib import Path

def print_banner():
    print("="*60)
    print(" CONTAINER CODE DETECTION SYSTEM ".center(60, "#"))
    print("="*60)

def main():
    while True:
        print_banner()
        print("1. [Dataset 1] - Prepare Block Data (Stage 1)")
        print("2. [Train 1]   - Train Block Detector (Stage 1)")
        print("3. [Utility]   - Extract detected blocks to 'data/character_labeling'")
        print("4. [Advanced]  - Train Custom OCR (Optional - CNN Sequence)")
        print("5. [Run All]   - Batch Pipeline (2-Stage EasyOCR)")
        print("6. [Test One]  - Single Test (Detailed View)")
        print("7. [Exit]")
        print("-" * 60)
        
        choice = input("Select an option (1-7): ").strip()
        
        if choice == '1':
            os.system("python -m src.training.prepare_container_data")
        elif choice == '2':
            os.system("python -m src.training.train_container --epochs 50 --batch 16 --imgsz 640")
        elif choice == '3':
            os.system("python -m src.training.crop_characters")
        elif choice == '4':
            os.system("python -m src.training.train_ocr --mode container")
        elif choice == '5':
            os.environ['DETECT_MODE'] = 'container'
            os.system("python -m src.inference.pipeline")
        elif choice == '6':
            path = input("Enter image path (or press Enter for default test image): ").strip()
            if not path:
                test_dir = Path("Container Code Detection.v1i.yolov8/test/images")
                if test_dir.exists():
                    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
                    if images:
                        path = str(images[0])
                    else:
                        print("No test images found.")
                        continue
                else:
                    print("Roboflow test directory not found.")
                    continue
            
            os.environ['DETECT_MODE'] = 'container'
            os.system(f"python -m src.inference.test_single {path}")
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")
        
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
