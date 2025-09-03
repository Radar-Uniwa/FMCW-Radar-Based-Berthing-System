# train_pointnet.py

import os
import glob
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import itertools  

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"base/path/to/data"  # Adjust as needed
SPLITS      = ["Training_2", "Validation_2", "Test_2"]# Adjust as needed based on the names 
NUM_POINTS  = 256
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.path.join(BASE_DIR, "name_of_model.pt") # Adjust as needed

# ─── DATASET ───────────────────────────────────────────────────────────────────
class RadarPointCloudDataset(Dataset):
    def __init__(self, base_dir, split, num_points):
        self.num_points = num_points
        pattern = os.path.join(base_dir, split, "*_labeled.csv")
        files = glob.glob(pattern)
        self.data = []
        for filepath in files:
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                pts = ast.literal_eval(row.get("Points","[]"))
                if not pts:
                    continue
                pts = np.array(pts, dtype=float)[:, :5]
                if pts.shape[0] >= num_points:
                    idx = np.random.choice(pts.shape[0], num_points, replace=False)
                    pts = pts[idx]
                else:
                    pad = np.zeros((num_points - pts.shape[0], pts.shape[1]), dtype=float)
                    pts = np.vstack([pts, pad])
                label = int(row["Dock_Label"])
                self.data.append((pts, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pts, label = self.data[idx]
        pts_tensor = torch.tensor(pts.T, dtype=torch.float32)
        return pts_tensor, torch.tensor(label, dtype=torch.long)

# ─── MODEL ─────────────────────────────────────────────────────────────────────
class PointNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1);       self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1);     self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512);          self.bn4 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256);           self.bn5 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]   # global max pool
        x = torch.relu(self.bn4(self.fc1(x))); x = self.dp1(x)
        x = torch.relu(self.bn5(self.fc2(x))); x = self.dp2(x)
        return self.fc3(x)

def main():
    # ─── LOAD & PREPARE DATA ─────────────────────────────────────────────────────
    loaders = {}
    for split in SPLITS:
        ds = RadarPointCloudDataset(BASE_DIR, split, NUM_POINTS)
        shuffle = (split == "Training_2")# Adjust as needed based on the line 23 
        # set num_workers=0 on Windows
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
        loaders[split] = loader
        print(f"{split}: {len(ds)} samples")

    # ─── CLASS WEIGHTS ─────────────────────────────────────────────────────────
    labels = [label for _, label in loaders["Training_2"].dataset.data]# Adjust as needed based on the line 23 
    counts = np.bincount(labels)
    total = sum(counts)
    class_weights = torch.tensor(
        [total/counts[0], total/counts[1]], dtype=torch.float32, device=DEVICE
    )

    # ─── MODEL, LOSS, OPTIM ──────────────────────────────────────────────────────
    model = PointNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ─── TRAIN/VAL LOOP ──────────────────────────────────────────────────────────
    best_f1, counter, patience = 0.0, 0, 10
    for epoch in range(1,101):
        model.train()
        for xb, yb in loaders["Training_2"]:# Adjust as needed based on the line 23 
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loaders["Validation_2"]:# Adjust as needed based on the line 23 
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy()
                preds.append(out.argmax(axis=1)); trues.append(yb.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        f1 = f1_score(trues, preds, average="macro")
        print(f"Epoch {epoch:03d} ▶ Val F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1, counter = f1, 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(" Saved best model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping."); break

    # ─── TEST EVAL & CURVES ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loaders["Test_2"]:# Adjust as needed based on the line 23 
            xb = xb.to(DEVICE)
            out = model(xb)
            probs.append(torch.softmax(out, dim=1)[:,1].cpu().numpy())
            preds.append(out.cpu().numpy().argmax(axis=1))
            trues.append(yb.numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues); probs = np.concatenate(probs)

    print("\n=== Test Metrics ===")
    print("Acc:", accuracy_score(trues, preds))
    print(classification_report(trues, preds, target_names=["No Dock","Dock"]))

    # ─── Matplotlib Confusion Matrix ────────────────────────────────────────────
    cm = confusion_matrix(trues, preds)
    classes = ["No Dock", "Dock"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)

    # Font sizes
    title_fs = 22
    label_fs = 18
    tick_fs  = 16
    annot_fs = 20

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.title.set_size(title_fs)
    ax.set_xticklabels(classes, fontsize=tick_fs)
    ax.set_yticklabels(classes, fontsize=tick_fs)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=annot_fs
        )

    plt.tight_layout()
    plt.show()

    # ─── ROC Curve ──────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(trues, probs); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.show()

    # ─── PR Curve ───────────────────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(trues, probs); ap = average_precision_score(trues, probs)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.legend(); plt.show()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
