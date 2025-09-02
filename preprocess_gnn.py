import os
import ast
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

# ─── Configuration ───────────────────────────────────────────────────────────────
BASE_PATH   = r"C:\Users\fotis\Downloads"
FOLDERS     = ["Training_2", "Validation_2", "Test_2"]
SCALER_PATH = os.path.join(BASE_PATH, "gnn_scaler.pkl")

# ─── Step 1: Fit scaler on all Training points ───────────────────────────────────
print("Gathering training points to fit scaler...")
all_pts = []
train_dir = os.path.join(BASE_PATH, "Training")
for fname in os.listdir(train_dir):
    if not fname.lower().endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(train_dir, fname))
    for s in df["Points"].dropna():
        pts = np.array(ast.literal_eval(s), dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        all_pts.append(pts)
all_pts = np.vstack(all_pts)  # shape (N_total, 5)
print(f"  Total training points: {all_pts.shape[0]}")

# Fit a StandardScaler on [x,y,z,doppler,snr]
scaler = StandardScaler().fit(all_pts)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to: {SCALER_PATH}\n")

# ─── Step 2: Preprocess each folder with the same scaler ────────────────────────
for folder in FOLDERS:
    folder_path = os.path.join(BASE_PATH, folder)
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    print(f"Processing {folder}...")
    scaler = joblib.load(SCALER_PATH)
    data_list = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".csv"):
            continue
        csv_path = os.path.join(folder_path, fname)
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            pts_str = row.get("Points", "")
            if pd.isna(pts_str) or pts_str == "[]":
                continue
            try:
                # parse raw points into (N,5)
                pts = np.array(ast.literal_eval(pts_str), dtype=float)
                if pts.ndim == 1:
                    pts = pts.reshape(1, -1)
                # normalize features
                pts_scaled = scaler.transform(pts)
                x = torch.tensor(pts_scaled, dtype=torch.float)
                # build kNN graph on full 5D feature space
                edge_index = knn_graph(x, k=20, loop=False)
                # graph label
                y = torch.tensor([int(row["Dock_Label"])], dtype=torch.long)
                data_list.append(Data(x=x, edge_index=edge_index, y=y))
            except Exception as e:
                print(f"  Skipping row {idx} in {fname}: {e}")

    out_path = os.path.join(BASE_PATH, f"{folder.lower()}_gnn.pt")
    torch.save(data_list, out_path)
    print(f"  Saved {len(data_list)} graphs to {out_path}\n")
