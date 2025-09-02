import ast
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection

# ─── Configuration ───────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\fotis\Downloads\Test_2\log_Lavrio_04_12-08_36_labeled.csv"

# ─── 1) Load CSV and build per-cluster feature matrix ────────────────────────────
df = pd.read_csv(CSV_PATH)

features = []
labels   = []
for _, row in df.iterrows():
    pts_str = row.get("Points", "")
    if pd.isna(pts_str) or pts_str == "[]":
        continue
    pts = np.array(ast.literal_eval(pts_str), dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    # cluster descriptor: mean of each column [x, y, z, doppler, snr]
    features.append(pts.mean(axis=0))
    labels.append(int(row["Dock_Label"]))

X = np.vstack(features)   # shape (n_clusters, 5)
y = np.array(labels)      # shape (n_clusters,)

# ─── 2) PCA → 3D embedding ───────────────────────────────────────────────────────
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y)
ax.set_title("PCA 3D Embedding")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.colorbar(sc, ax=ax, label="Dock Label")

# ─── 3) UMAP → 3D embedding ──────────────────────────────────────────────────────
reducer = umap.UMAP(n_components=3, random_state=42)
X_umap = reducer.fit_transform(X)

fig2 = plt.figure()
ax2  = fig2.add_subplot(111, projection='3d')
sc2  = ax2.scatter(X_umap[:,0], X_umap[:,1], X_umap[:,2], c=y)
ax2.set_title("UMAP 3D Embedding")
ax2.set_xlabel("UMAP1")
ax2.set_ylabel("UMAP2")
ax2.set_zlabel("UMAP3")
plt.colorbar(sc2, ax=ax2, label="Dock Label")

plt.show()
