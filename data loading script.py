import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


main_path = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008\ImageSets\Main"

# List of 20 VOC classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Dictionary to hold image IDs and their label vectors
image_labels = {}

# Loop through all 20 classes
for idx, cls in enumerate(classes):
    file_path = os.path.join(main_path, f"{cls}_trainval.txt")
    with open(file_path, 'r') as f:
        for line in f:
            image_id, label = line.strip().split()
            label = 1 if int(label) == 1 else 0  # -1 or 0 = not present
            if image_id not in image_labels:
                image_labels[image_id] = np.zeros(len(classes), dtype=int)
            image_labels[image_id][idx] = label

# Create a list of (image_id, label_vector) pairs
trainval_data = [(img_id, label) for img_id, label in image_labels.items()]

print(f"Total samples loaded: {len(trainval_data)}")

# Path to JPEG images
image_path = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008\JPEGImages"

# Define a custom dataset
class VOCDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data  # list of (image_id, label_vector)
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

# Define transforms (resize + tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Create dataset and DataLoader
dataset = VOCDataset(trainval_data, image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test: Get one batch
images, labels = next(iter(dataloader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
