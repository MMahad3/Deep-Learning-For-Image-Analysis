import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, multilabel_confusion_matrix, average_precision_score
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOC classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Load labels
main_path = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008\ImageSets\Main"
image_labels = {}
for idx, cls in enumerate(classes):
    with open(os.path.join(main_path, f"{cls}_trainval.txt"), 'r') as f:
        for line in f:
            image_id, label = line.strip().split()
            label = 1 if int(label) == 1 else 0
            if image_id not in image_labels:
                image_labels[image_id] = np.zeros(len(classes), dtype=int)
            image_labels[image_id][idx] = label
trainval_data = [(img_id, label) for img_id, label in image_labels.items()]

# Dataset
image_path = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008\JPEGImages"

class VOCDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, label = self.data[idx]
        img_file = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create DataLoader
dataset = VOCDataset(trainval_data, image_path, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 20)
model = model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training complete")

# Evaluation
model.eval()
all_labels = []
all_outputs = []
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())

all_outputs = np.vstack(all_outputs)
all_labels = np.vstack(all_labels)

# Binarize outputs at threshold 0.5
preds = (all_outputs >= 0.5).astype(int)

# Metrics
print("\n--- Classification Report ---")
print(classification_report(all_labels, preds, target_names=classes))

# Confusion Matrix
conf_matrix = multilabel_confusion_matrix(all_labels, preds)
for i, cm in enumerate(conf_matrix):
    print(f"\nConfusion Matrix for class '{classes[i]}':\n{cm}")

# Mean Average Precision
mAP = average_precision_score(all_labels, all_outputs, average="macro")
print(f"\nMean Average Precision (mAP): {mAP:.4f}")

# IoU calculation (per class)
def iou_score(y_true, y_pred):
    intersection = (y_true & y_pred).sum(axis=0)
    union = (y_true | y_pred).sum(axis=0)
    return (intersection / np.clip(union, a_min=1, a_max=None))

ious = iou_score(all_labels.astype(bool), preds.astype(bool))
for iou_val, cls in zip(ious, classes):
    print(f"IoU for {cls}: {iou_val:.4f}")
print(f"Mean IoU: {ious.mean():.4f}")
