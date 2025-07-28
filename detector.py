import cv2
from ultralytics import YOLO
import easyocr
import json
from pathlib import Path

class YOLOLayoutExtractor:
    def __init__(self, model_path, langs=['fr', 'en'], confidence=0.5):
        """
        Initialise le pipeline YOLOv8 + OCR.
        model_path: chemin vers le modèle YOLOv8 (.pt)
        langs: langues pour EasyOCR
        confidence: seuil de confiance pour les détections
        """
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(langs)
        self.confidence = confidence

    def load_image(self, image_path):
        """Charge une image depuis un chemin."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        return image

    def detect_layout(self, image):
        """
        Détecte les zones avec YOLOv8.
        Retourne une liste de dicts avec type, bbox, confiance.
        """
        results = self.model(image)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            label = self.model.names[cls_id]
            conf = float(box.conf)
            if conf < self.confidence:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "type": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
        return detections

    def extract_text_from_zone(self, image, bbox):
        """Extrait le texte d'une zone image+bbox avec EasyOCR."""
        x1, y1, x2, y2 = bbox
        zone = image[y1:y2, x1:x2]
        result = self.reader.readtext(zone, detail=0)
        return " ".join(result)

    def process_image(self, image_path):
        """
        Pipeline complet :
        1. Charge l'image
        2. Détecte les zones
        3. Extrait le texte pour chaque zone
        4. Retourne une structure JSON
        """
        image = self.load_image(image_path)
        zones = self.detect_layout(image)
        output = []

        for zone in zones:
            text = self.extract_text_from_zone(image, zone["bbox"])
            zone["text"] = text
            output.append(zone)

        return output

    def save_result(self, data, output_path="result.json"):
        """Sauvegarde les résultats en JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Résultats sauvegardés dans {output_path}")