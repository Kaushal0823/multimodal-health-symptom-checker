# ============================================================
# HAM10000 SKIN LESION CNN — FINAL TRAINING SCRIPT
# Naye Colab notebook mein use karo. Runtime -> T4 GPU set karo.
# Har CELL ko alag Colab cell mein copy-paste karo, order mein.
# ============================================================

# ---------------- CELL 1: Drive mount ----------------
from google.colab import drive
drive.mount('/content/drive')

# ---------------- CELL 2: Images unzip karo ----------------
!mkdir -p /content/images
!unzip -q "/content/drive/MyDrive/health_project/cleaned_images.zip" -d /content/images
!ls /content/images | head
!echo "Total files:"
!find /content/images -name "*.jpg" | wc -l

# ---------------- CELL 3: Libraries ----------------
!pip install -q torch torchvision scikit-learn pandas pillow tqdm

# ---------------- CELL 4: Metadata load + images se match karo ----------------
import pandas as pd
import os

METADATA_PATH = "/content/drive/MyDrive/health_project/HAM10000_metadata.csv"

IMAGE_DIR = "/content/images/cleaned_images"
if not os.path.exists(IMAGE_DIR):
    IMAGE_DIR = "/content/images"
print("Using IMAGE_DIR:", IMAGE_DIR)

df = pd.read_csv(METADATA_PATH)
print("Total metadata rows:", len(df))
print(df['dx'].value_counts())

available_images = set(os.listdir(IMAGE_DIR))
df['filename'] = df['image_id'] + '.jpg'
df = df[df['filename'].isin(available_images)].reset_index(drop=True)
print("Rows after matching with cleaned images:", len(df))

# ---------------- CELL 5: Label encode + train/val split ----------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['dx'])
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

with open('/content/drive/MyDrive/health_project/image_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ---------------- CELL 6: Class weights ----------------
import numpy as np
import torch

class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

# ---------------- CELL 7: Dataset class + augmentation ----------------
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row['filename']}"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['label']

train_dataset = SkinDataset(train_df, IMAGE_DIR, transform=train_transform)
val_dataset = SkinDataset(val_df, IMAGE_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print("Train batches:", len(train_loader), "| Val batches:", len(val_loader))

# ---------------- CELL 8: Model (ResNet50, frozen backbone) ----------------
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# ---------------- CELL 9: Training loop, WITH per-epoch checkpointing ----------------
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

CHECKPOINT_DIR = "/content/drive/MyDrive/health_project/image_model_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, all_preds, all_labels

NUM_EPOCHS = 10
best_f1 = 0.0

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt"))

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        print(f"  -> new best model saved (f1={val_f1:.4f})")

# ---------------- CELL 9b: Resume after a disconnect (use INSTEAD of Cell 9 if resuming) ----------------
# import os
# print(sorted(os.listdir(CHECKPOINT_DIR)))
# resume_epoch = 6
# model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"epoch_{resume_epoch}.pt")))
# print(f"Resumed from epoch {resume_epoch}")
# # then re-run Cell 9's loop with: for epoch in range(resume_epoch, NUM_EPOCHS):

# ---------------- CELL 10: Final per-class evaluation ----------------
from sklearn.metrics import classification_report, confusion_matrix

model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt")))
_, _, final_preds, final_labels = evaluate(model, val_loader, device)

print(classification_report(final_labels, final_preds, target_names=label_encoder.classes_))
print("Confusion matrix:")
print(confusion_matrix(final_labels, final_preds))

# ---------------- CELL 11: Final model save (Drive, permanent) ----------------
final_path = "/content/drive/MyDrive/health_project/final_image_model_resnet50.pth"
torch.save(model.state_dict(), final_path)
print("Final model saved to:", final_path)
