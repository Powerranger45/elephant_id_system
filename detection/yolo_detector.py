from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import os
import rawpy

class ElephantDetector:
    def __init__(self, model_path="models/fusionnet_best.pth"):
        """Initialize YOLOv8 and trained FusionNet"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO("models/yolo_best.pt")

        # Load trained FusionNet
        self.fusion_model = self._load_fusionnet(model_path)
        self.known_embeddings = {}
        self.next_id = 0

    def _load_fusionnet(self, path):
        """Load trained FusionNet model"""
        model = torch.load(path, map_location=self.device)
        model.eval()
        return model.to(self.device)

    def _extract_ear(self, image):
        """Detect elephant and extract ear region"""
        results = self.yolo(image, verbose=False)
        ear_crops = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ear_w = int((x2 - x1) * 0.3)
                ear_h = int((y2 - y1) * 0.3)
                ear_crops.append(cv2.cvtColor(image[y1:y1+ear_h, x1:x1+ear_w], cv2.COLOR_BGR2RGB))

        return ear_crops

    def _get_embedding(self, image):
        """Get embedding using FusionNet"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.fusion_model(tensor)
            _, pred = torch.max(outputs, 1)
        return pred.item()

    def identify(self, image_path):
        """Identify elephants in image"""
        if image_path.lower().endswith('.nrw'):
            with rawpy.imread(image_path) as raw:
                image = raw.postprocess()
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect elephants and extract ears
        ear_crops = self._extract_ear(image)
        results = []

        for crop in ear_crops:
            # Get prediction from FusionNet
            prediction = self._get_embedding(crop)
            results.append({
                "id": f"Elephant_{prediction:03d}",
                "score": 0.95  # Placeholder for similarity score
            })

        return results
