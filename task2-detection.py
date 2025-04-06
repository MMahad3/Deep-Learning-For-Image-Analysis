import os
import xml.etree.ElementTree as ET
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from PIL import Image

# Dataset Class
class VOCDataset(Dataset):
    def __init__(self, root, image_set="train", transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_ids = open(os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")).read().splitlines()
        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id + ".jpg")).convert("RGB")
        ann_path = os.path.join(self.ann_dir, img_id + ".xml")
        tree = ET.parse(ann_path)

        boxes = []
        labels = []
        for obj in tree.findall("object"):
            label = obj.find("name").text
            labels.append(self.class_names.index(label) + 1)  # Background = 0
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Optionally round them to nearest integer, or cast to int directly
        box = [round(xmin), round(ymin), round(xmax), round(ymax)]  # You can replace round() with int() if you prefer
        
        boxes.append(box)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.image_ids)

# Model Setup
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Collate Function
def collate_fn(batch):
    return tuple(zip(*batch))

# Training
def train_model(model, train_loader, device, epochs=3, save_path=None):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        total_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# Evaluation
def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []
    iou_total, count = 0.0, 0
    map_metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, pred in zip(targets, outputs):
                true_labels = target["labels"].numpy()
                pred_labels = pred["labels"].cpu().numpy()
                y_true.extend(true_labels)
                y_pred.extend(pred_labels)

                iou_total += box_iou(pred["boxes"].cpu(), target["boxes"]).mean().item()
                count += 1

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in outputs]
            targets_cpu = [{k: v for k, v in t.items()} for t in targets]
            map_metric.update(preds_cpu, targets_cpu)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nMean IoU: {iou_total / count:.4f}")
    print(f"\nmAP Results:")
    print(map_metric.compute())

# Main Function
def main():
    root = r"C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008"  # Update if your folder name is different
    num_classes = 21  # 20 classes + background
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VOCDataset(root=root, image_set="train", transforms=F.to_tensor)
    val_dataset = VOCDataset(root=root, image_set="val", transforms=F.to_tensor)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes)

    # Training the model
    model_save_path = "faster_rcnn.pth"
    train_model(model, train_loader, device, epochs=3, save_path=model_save_path)

    # Evaluate the model after training
    evaluate_model(model, val_loader, device)

    # Optionally load the model and continue evaluation or inference:
    # model.load_state_dict(torch.load(model_save_path))

if __name__ == "__main__":
    main()
