import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ElephantEarDataset(Dataset):
    def __init__(self, data_dir, yolo_model, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.yolo = yolo_model
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect images
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.nrw')):
                    self.image_paths.append(os.path.join(cls_dir, img_file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def _detect_ear(self, image):
        """Detect elephant and extract ear region"""
        results = self.yolo(image, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ear_w = int((x2 - x1) * 0.3)
                ear_h = int((y2 - y1) * 0.3)
                return image[y1:y1+ear_h, x1:x1+ear_w]
        return image  # Fallback to full image

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Handle RAW files
            if img_path.lower().endswith('.nrw'):
                with rawpy.imread(img_path) as raw:
                    image = raw.postprocess()
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract ear region using YOLO
            ear_crop = self._detect_ear(image)
            ear_image = Image.fromarray(ear_crop)

            if self.transform:
                ear_image = self.transform(ear_image)

            return ear_image, label

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label

class FusionNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(FusionNet, self).__init__()
        # Custom CNN Stream
        self.cnn_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # EfficientNet Stream
        self.effnet = models.efficientnet_b0(pretrained=True)
        self.effnet.classifier = nn.Identity()
        # Feature Fusion Module
        self.fusion = nn.Sequential(
            nn.Linear(256 + 1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn_stream(x).flatten(1)
        eff_features = self.effnet(x)
        return self.fusion(torch.cat((cnn_features, eff_features), dim=1))

def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate Sensitivity, Specificity, and TSS"""
    cm = confusion_matrix(y_true, y_pred)
    tss_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss_scores.append(sensitivity + specificity - 1)

    return np.mean(tss_scores)

def train_model():
    # Initialize YOLOv8
    yolo_model = YOLO("models/yolo_best.pt")

    # Dataset transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = ElephantEarDataset("datasets/flat_dataset", yolo_model, transform=transform)
    num_classes = len(dataset.classes)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = FusionNet(num_classes).to(device)
    model.effnet.requires_grad_(False)  # Freeze initially

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        tss = calculate_metrics(all_labels, all_preds, num_classes)

        print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, TSS: {tss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/fusionnet_best.pth")

    return model

if __name__ == "__main__":
    train_model()
