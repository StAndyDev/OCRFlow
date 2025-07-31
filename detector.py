# detector.py
import cv2
from ultralytics import YOLO
import json
import time

class YOLOLayoutDetector:
    def __init__(self, model_path, confidence=0.5):
        """
        Initialise le modèle YOLOv8 pour la détection de mise en page.
        model_path: chemin vers le modèle YOLOv8 (.pt)
        confidence: seuil de confiance pour filtrer les détections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def load_image(self, image_path):
        """Charge une image depuis un chemin donné."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        return image

    def detect_layout(self, image):
        """
        Détecte les éléments de mise en page avec YOLOv8.
        Retourne une liste de dictionnaires avec type, bbox, et confiance.
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

    def process_image(self, image_path):
        """
        Charge une image et détecte les éléments de mise en page.
        Retourne une liste de zones détectées.
        """
        image = self.load_image(image_path)
        zones = self.detect_layout(image)
        return zones

    def detect_from_camera(self, camera_id=0, fps=5, resize_width=640, resize_height=480, skip_frames=3):
        """
        Active la webcam pour une détection en temps réel des mises en page avec YOLOv8.
        
        Paramètres :
        - camera_id : ID de la webcam (0 par défaut)
        - fps : nombre d'images traitées par seconde (ex: 5)
        - resize_width / resize_height : dimensions pour redimensionner les frames (640x480 recommandé)
        - skip_frames : nombre de frames à sauter avant chaque détection (ex: 3 = une détection toutes les 4 frames)

        Appuie sur 'q' pour quitter.
        """
        import time

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Erreur: Impossible d'accéder à la caméra.")
            return

        print(f"Détection en temps réel lancée (fps cible: {fps}) - Appuie sur 'q' pour quitter.")
        delay = 1.0 / fps
        frame_count = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire l'image depuis la caméra.")
                break

            # Réduction de la taille pour accélérer le traitement
            frame = cv2.resize(frame, (resize_width, resize_height))
            frame_count += 1

            if frame_count % (skip_frames + 1) == 0:
                detections = self.detect_layout(frame)

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = f'{det["type"]} ({det["confidence"]:.2f})'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Layout Detection", frame)

            elapsed = time.time() - start_time
            time_to_wait = delay - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()




    def save_result(self, data, output_path="result.json"):
        """Sauvegarde les résultats de détection au format JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Résultats sauvegardés dans {output_path}")