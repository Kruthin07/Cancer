import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from model import CancerCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.0001
VAL_SPLIT = 0.2

# --------- TRANSFORMS (RESNET REQUIRED) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASETS ----------------
DATASETS = {
    "oral": "../dataset/oral/train",
    "eye": "../dataset/eye",
    "skin": "../dataset/skin"
}

os.makedirs("models", exist_ok=True)

# ---------------- TRAIN LOOP ----------------
for cancer_type, path in DATASETS.items():
    print(f"\n==============================")
    print(f" Training {cancer_type.upper()} CANCER MODEL")
    print(f"==============================")

    # ---- Safety checks ----
    for cls in ["cancer", "normal"]:
        folder = os.path.join(path, cls)
        if not os.path.exists(folder) or len(os.listdir(folder)) == 0:
            raise RuntimeError(f"❌ Missing images in {folder}")

    # ---- Load full dataset ----
    full_dataset = datasets.ImageFolder(path, transform=transform)
    class_to_idx = full_dataset.class_to_idx
    if "cancer" not in class_to_idx or "normal" not in class_to_idx:
        raise RuntimeError(
            f"Expected classes 'cancer' and 'normal' in {path}, found {list(class_to_idx.keys())}"
        )
    cancer_idx = class_to_idx["cancer"]

    # Stratified train/val split using binary labels: 1 = cancer, 0 = normal
    all_targets = np.array(full_dataset.targets)
    y_bin = (all_targets == cancer_idx).astype(int)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=42
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(y_bin)), y_bin))
    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset = Subset(full_dataset, val_idx.tolist())

    # Balance classes during training (helps a lot if dataset is imbalanced)
    y_train = y_bin[train_idx]
    class_counts = np.bincount(y_train, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    model = CancerCNN(trainable_layers=2, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ---- Epochs ----
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for images, labels in train_loader:
            images = images.to(device)
            # Binary target: 1 = cancer, 0 = normal
            labels_bin = (labels == cancer_idx).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_bin)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # -------- VALIDATION --------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images).squeeze(1)

                preds = (torch.sigmoid(outputs) > 0.5).long()
                labels_bin = (labels == cancer_idx).long()
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_labels.extend(labels_bin.detach().cpu().numpy().tolist())

        # ---- Metrics ----
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)
        scheduler.step(acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {sum(train_losses)/len(train_losses):.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Prec: {prec:.4f} | "
            f"Recall: {rec:.4f} | "
            f"F1: {f1:.4f}"
        )

    # ---- Save model ----
    torch.save(model.state_dict(), f"models/{cancer_type}_model.pth")
    print(f"✅ Saved models/{cancer_type}_model.pth")
