from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import rawpy

class ElephantDetector:
    def __init__(self, yolo_path="models/yolo_best.pt"):
        """Initialize YOLOv8 detector"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(yolo_path)
        self.known_embeddings = {}
        self.next_id = 0

    def detect(self, image):
        """Detect elephants and extract ear regions"""
        results = self.model(image, verbose=False)
        ear_crops = []
        bboxes = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))

                # Extract ear region (20% from top-left corner)
                ear_w = int((x2 - x1) * 0.3)
                ear_h = int((y2 - y1) * 0.3)

                ear_crop = image[y1:y1+ear_h, x1:x1+ear_w]
                if ear_crop.size > 0:
                    ear_crops.append(cv2.cvtColor(ear_crop, cv2.COLOR_BGR2RGB))

        return ear_crops, bboxes
