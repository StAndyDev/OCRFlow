import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import YOLOLayoutExtractor

def main():
    image_path = "data/raw/IMG_20190709_080910.jpg"
    extractor = YOLOLayoutExtractor(model_path="malaysia-aiYOLOv8X-DocLayNet-Full-1024-42.pt")
    results = extractor.process_image(image_path)
    extractor.save_result(results)

    # Étape 1 : prétraitement
    # Étape 2 : détection des zones
    # Étape 3 : OCR sur chaque zone
    # Étape 4 : Structuration
    # Export ou affichage

if __name__ == "__main__":
    main()