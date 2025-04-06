from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torch
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Define image transformations 
input_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
])


# Define a transform to convert PIL Image to PyTorch tensor for the target
target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.PILToTensor()  
])


# Datast
voc_root = 'C:\FAST UNIVERSITY\8th semester\DLP\DLP MID 2\TASK 1-2-3\VOCtrainval_14-Jul-2008\VOCdevkit\VOC2008'


dataset = VOCSegmentation(root=voc_root, year='2008', image_set='val', 
                          download=True, transform=input_transforms,
                          target_transform=target_transform) 


dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)


# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval() 




# Function to visualize the predicted and actual segmentation masks
def visualize_prediction(image, mask, pred_mask):
    # Convert tensor to numpy arrays
    image = image.permute(1, 2, 0).cpu().numpy()  
    mask = mask.squeeze(0).cpu().numpy()  
    pred_mask = pred_mask.cpu().numpy()
    
    # Plot the image and its segmentation masks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    
    axes[1].imshow(mask)
    axes[1].set_title('Ground Truth Mask')
    
    axes[2].imshow(pred_mask)
    axes[2].set_title('Predicted Mask')
    
    for ax in axes:
        ax.axis('off')
    
    plt.show()

# Iterate over the dataset to perform inference
for images, targets in dataloader:
    images = images.to(device)  
    targets = targets.to(device)  

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(images)['out']  
    
    
    pred_mask = torch.argmax(outputs, dim=1)  
    
    # Visualize the result 
    visualize_prediction(images[0], targets[0], pred_mask[0])
    break  




def calculate_metrics(pred_mask, targets, num_classes):
    # Flatten the masks
    pred_mask = pred_mask.view(-1)
    targets = targets.view(-1)

    # Calculate Accuracy
    accuracy = accuracy_score(targets.cpu().numpy(), pred_mask.cpu().numpy())

    # Calculate Precision, Recall, F1-score for each class
    precision = precision_score(targets.cpu().numpy(), pred_mask.cpu().numpy(), average=None, labels=np.arange(num_classes))
    recall = recall_score(targets.cpu().numpy(), pred_mask.cpu().numpy(), average=None, labels=np.arange(num_classes))
    f1 = f1_score(targets.cpu().numpy(), pred_mask.cpu().numpy(), average=None, labels=np.arange(num_classes))

    # Confusion Matrix
    cm = confusion_matrix(targets.cpu().numpy(), pred_mask.cpu().numpy(), labels=np.arange(num_classes))

    # Calculate IoU (Intersection over Union) for each class
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))

    return accuracy, precision, recall, f1, cm, iou


num_classes = 21  


all_predictions = []
all_targets = []

for images, targets in dataloader:
    images = images.to(device)  
    targets = targets.to(device)  

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(images)['out']  
    
    # Get the predicted segmentation mask (the class with the highest score)
    pred_mask = torch.argmax(outputs, dim=1)  #

    # Collect predictions and targets for evaluation
    all_predictions.append(pred_mask)
    all_targets.append(targets)

# Flatten all predictions and targets for final evaluation
all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Calculate evaluation metrics
accuracy, precision, recall, f1, cm, iou = calculate_metrics(all_predictions, all_targets, num_classes)

# Print the evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision per class: {precision}')
print(f'Recall per class: {recall}')
print(f'F1-score per class: {f1}')

# Confusion Matrix
print('Confusion Matrix:')
print(cm)



mAP = np.mean(precision)
print(f'mAP: {mAP}')

# IoU per class
print(f'IoU per class: {iou}')
