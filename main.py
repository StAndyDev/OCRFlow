import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import YOLOLayoutDetector

def main():
    # --- Config ---
    model_path = "malaysia-aiYOLOv8X-DocLayNet-Full-1024-42.pt"
    image_path = "data/raw/image-5bd08790e1864.png"
    output_json = "result.json"

    # --- Init du détecteur ---
    detector = YOLOLayoutDetector(model_path, confidence=0.5)

    # --- Traitement ---
    zones = detector.process_image(image_path)
    print(f"Zones détectées : {len(zones)}")
    # --- Affichage console ---
    for i, zone in enumerate(zones):
        print(f"Zone {i+1}:")
        print(f"  Type: {zone['type']}")
        print(f"  Confiance: {zone['confidence']:.2f}")
        print(f"  BBox: {zone['bbox']}")
        print("")

    # --- Sauvegarde JSON (optionnel) ---
    detector.save_result(zones, output_json)

    # Étape 1 : prétraitement
    # Étape 2 : détection des zones
    # Étape 3 : OCR sur chaque zone
    # Étape 4 : Structuration
    # Export ou affichage

if __name__ == "__main__":
    main()