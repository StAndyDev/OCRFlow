# detector_doclayout.py
import cv2
import json
from doclayout_yolo import YOLOv10

class DocLayoutDetector:
    def __init__(self, model_path=None, confidence=0.5, device="cpu"):
        """
        Initialise DocLayout‑YOLO pour la détection de mise en page.
        model_path: chemin vers le modèle .pt ou None pour charger depuis Huggingface.
        confidence: seuil de confiance pour filtrer les détections.
        device: "cpu" ou "cuda:0"
        """
        if model_path:
            self.model = YOLOv10(model_path)
        else:
            self.model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
        self.confidence = confidence
        self.device = device

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        return image

    def detect_layout(self, image_path, imgsz=1024):
        """
        Détecte les éléments de mise en page avec DocLayout‑YOLO.
        Retourne une liste de dict {type, bbox, confidence}.
        """
        det_res = self.model.predict(image_path, imgsz=imgsz,
                                     conf=self.confidence, device=self.device)
        detections = []
        # traitement du premier résultat (batch 1)
        for det in det_res[0].boxes:
            cls_id = int(det.cls)
            label = det_res[0].names[cls_id]
            conf = float(det.conf)
            if conf < self.confidence:
                continue
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            detections.append({
                "type": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
        return detections

    def process_image(self, image_path, imgsz=1024):
        image = self.load_image(image_path)
        zones = self.detect_layout(image_path, imgsz=imgsz)
        return zones

    def save_result(self, data, output_path="result.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Résultats sauvegardés dans {output_path}")