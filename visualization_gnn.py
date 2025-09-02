import sys
import ast
import joblib
import cv2
import torch
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QPointF
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, GCNConv, global_mean_pool
import torch.nn.functional as F
import itertools  

def parse_time_to_sec(tstr: str) -> float:
    parts = [float(p) for p in tstr.split(':')]
    if len(parts) == 3:
        h, m, s = parts; return h*3600 + m*60 + s
    if len(parts) == 2:
        m, s = parts; return m*60 + s
    return parts[0]

class RadarGCN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels, 2)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class DockVisualizer:
    def __init__(self, csv_path, video_path, model_path, scaler_path):
        # ─── Load & parse CSV ─────────────────────────────────────────
        df = pd.read_csv(csv_path)
        df['SecAbs'] = df['Time'].apply(parse_time_to_sec)
        base = df['SecAbs'].min()
        df['SecRel'] = (df['SecAbs'] - base).astype(int)
        df['Points'] = df['Points'].fillna('[]').astype(str).str.strip()
        df['Points'] = df['Points'].apply(lambda s: ast.literal_eval(s) if s else [])
        self.df = df.reset_index(drop=True)

        # build point list
        self.points_list = []
        for pts in self.df['Points']:
            arr = np.array(pts, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.points_list.append(arr)

        # group by relative second
        self.sec2idx = self.df.groupby('SecRel').indices
        self.max_sec = max(self.sec2idx.keys())

        # ─── Load scaler & model ─────────────────────────────────────
        self.scaler = joblib.load(scaler_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = RadarGCN(in_channels=5, hidden_channels=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Precompute cluster predictions
        self.cluster_results = []
        for pts in self.points_list:
            if pts.shape[0] < 3:
                self.cluster_results.append((None, 0.0))
            else:
                self.cluster_results.append(self.classify_cluster(pts))

        # per-second lookup
        self.sec2clusters = {
            sec: [(i, self.cluster_results[i]) for i in idxs]
            for sec, idxs in self.sec2idx.items()
        }

        # precompute distance_by_sec
        self.distance_by_sec = {}
        for sec, pairs in self.sec2clusters.items():
            ys = [ self.points_list[i][:,1].mean()
                   for i,(cls,_) in pairs if cls==1 ]
            self.distance_by_sec[sec] = float(min(ys)) if ys else None

        # ─── Build UI ─────────────────────────────────────────────────
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("Dock Distance Visualization")
        layout = QtWidgets.QGridLayout(self.win)

        # Side view (X–Y)
        self.xy_plot = pg.PlotWidget(background='#f7f7f7')
        self.xy_plot.setTitle('<span style="font-size:16pt">Side View (X–Y)</span>')
        self.xy_plot.setLabel('bottom','<span style="font-size:12pt">Y (m)</span>')
        self.xy_plot.setLabel('left',  '<span style="font-size:12pt">X (m)</span>')
        self.xy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.xy_plot.setXRange(0,60); self.xy_plot.setYRange(-10,10)
        self.xy_legend = self.xy_plot.addLegend(offset=(10,10))

        self.xy_scatter_dock    = pg.ScatterPlotItem(symbol='o', size=7,
                                                      brush=pg.mkBrush(255,0,0,200))
        self.xy_plot.addItem(self.xy_scatter_dock)
        self.xy_legend.addItem(self.xy_scatter_dock, 'Dock')

        self.xy_scatter_nondock = pg.ScatterPlotItem(symbol='o', size=7,
                                                      brush=pg.mkBrush(0,0,255,200))
        self.xy_plot.addItem(self.xy_scatter_nondock)
        self.xy_legend.addItem(self.xy_scatter_nondock, 'Non-dock')

        layout.addWidget(self.xy_plot, 0, 0)

        # Top view (Y–Z)
        self.yz_plot = pg.PlotWidget(background='#f7f7f7')
        self.yz_plot.setTitle('<span style="font-size:16pt">Top View (Y–Z)</span>')
        self.yz_plot.setLabel('bottom','<span style="font-size:12pt">Z (m)</span>')
        self.yz_plot.setLabel('left',  '<span style="font-size:12pt">Y (m)</span>')
        self.yz_plot.showGrid(x=True, y=True, alpha=0.3)
        self.yz_plot.setXRange(-10,10); self.yz_plot.setYRange(0,60)
        self.yz_legend = self.yz_plot.addLegend(offset=(10,10))

        self.yz_scatter_dock    = pg.ScatterPlotItem(symbol='o', size=7,
                                                      brush=pg.mkBrush(255,0,0,200))
        self.yz_plot.addItem(self.yz_scatter_dock)
        self.yz_legend.addItem(self.yz_scatter_dock, 'Dock')

        self.yz_scatter_nondock = pg.ScatterPlotItem(symbol='o', size=7,
                                                      brush=pg.mkBrush(0,0,255,200))
        self.yz_plot.addItem(self.yz_scatter_nondock)
        self.yz_legend.addItem(self.yz_scatter_nondock, 'Non-dock')

        layout.addWidget(self.yz_plot, 0, 1)

        # Video display (800×400) at 20 fps
        self.cap = cv2.VideoCapture(video_path)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(800, 400)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.video_label, 1, 0)

        # Controls + elapsed time
        ctrls = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("⏸ Pause")
        self.btn.setFixedWidth(100)
        self.btn.clicked.connect(self.toggle_play)
        ctrls.addWidget(self.btn)
        self.time_label = QtWidgets.QLabel("Elapsed: 00:00")
        self.time_label.setFont(QFont('',12))
        ctrls.addWidget(self.time_label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.max_sec)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.on_slider_change)
        ctrls.addWidget(self.slider, 1)
        layout.addLayout(ctrls, 2, 0, 1, 2)

        # Distance & schematic
        info = QtWidgets.QWidget(); vbox = QtWidgets.QVBoxLayout(info)
        self.dist_label = QtWidgets.QLabel("Distance to dock: N/A")
        self.dist_label.setFont(QFont('',24,QFont.Bold))
        vbox.addWidget(self.dist_label)
        self.diagram_label = QtWidgets.QLabel(); self.diagram_label.setFixedSize(400,100)
        vbox.addWidget(self.diagram_label); vbox.addStretch()
        layout.addWidget(info, 1, 1)

        # Prepare confidence labels
        self.text_items_xy = []
        self.text_items_yz = []

        # Playback state
        self.is_running     = True
        self.frame_idx      = 0
        self.video_fps      = 20
        self.current_second = 0

        # Timer: video at 20 fps
        self.video_timer = QtCore.QTimer(self.win)
        self.video_timer.timeout.connect(self.update_video)
        self.video_timer.start(int(1000 / self.video_fps))

        # Timer: overlay at 1 fps
        self.overlay_timer = QtCore.QTimer(self.win)
        self.overlay_timer.timeout.connect(self.update_overlay)
        self.overlay_timer.start(1000)

    def classify_cluster(self, pts):
        pts_scaled = self.scaler.transform(pts)
        x = torch.from_numpy(pts_scaled).float().to(self.device)
        k = max(1, min(20, x.size(0)-1))
        ei = knn_graph(x, k=k, loop=False)
        b = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        data = Data(x=x, edge_index=ei, batch=b).to(self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.batch)
            probs = F.softmax(out, dim=1)
        conf, cls = probs.max(dim=1)
        return int(cls.item()), float(conf.item())

    def draw_diagram(self, distance_m: float):
        # ... same as before ...
        pass

    def toggle_play(self):
        if self.is_running:
            self.video_timer.stop()
            self.overlay_timer.stop()
            self.btn.setText("▶ Play")
        else:
            self.video_timer.start(int(1000/self.video_fps))
            self.overlay_timer.start(1000)
            self.btn.setText("⏸ Pause")
        self.is_running = not self.is_running

    def on_slider_change(self, sec):
        # pause
        if self.is_running:
            self.toggle_play()
        # seek both streams
        self.current_second = sec
        self.frame_idx = sec * self.video_fps
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.update_video()
        self.update_overlay()

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.video_timer.stop()
            return
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)
        self.frame_idx += 1

    def update_overlay(self):
        sec = int(self.frame_idx / self.video_fps)
        if sec > self.max_sec:
            self.overlay_timer.stop()
            return
        # time label
        m, s = divmod(sec, 60)
        self.time_label.setText(f"Elapsed: {m:02d}:{s:02d}")
        # slider
        self.slider.blockSignals(True)
        self.slider.setValue(sec)
        self.slider.blockSignals(False)
        # clear old items
        for ti in self.text_items_xy: self.xy_plot.removeItem(ti)
        for ti in self.text_items_yz: self.yz_plot.removeItem(ti)
        self.text_items_xy.clear(); self.text_items_yz.clear()
        # plot clusters & confidence for this second
        dock_pts, non_pts = [], []
        dock_centroids, non_centroids = [], []
        dock_confs, non_confs = [], []
        for idx,(cls,conf) in self.sec2clusters.get(sec, []):
            if cls is None: continue
            pts = self.points_list[idx]
            cen = pts.mean(axis=0)
            if cls==1:
                dock_pts.append(pts); dock_centroids.append(cen); dock_confs.append(conf)
            else:
                non_pts.append(pts); non_centroids.append(cen); non_confs.append(conf)
        if dock_pts:
            dp = np.vstack(dock_pts)
            self.xy_scatter_dock.setData(x=dp[:,1], y=dp[:,0])
            self.yz_scatter_dock.setData(x=dp[:,2], y=dp[:,1])
        else:
            self.xy_scatter_dock.setData([],[]); self.yz_scatter_dock.setData([],[])
        if non_pts:
            npd = np.vstack(non_pts)
            self.xy_scatter_nondock.setData(x=npd[:,1], y=npd[:,0])
            self.yz_scatter_nondock.setData(x=npd[:,2], y=npd[:,1])
        else:
            self.xy_scatter_nondock.setData([],[]); self.yz_scatter_nondock.setData([],[])
        # Confidence labels (bigger, bold)
        for c, conf in zip(dock_centroids, dock_confs):
            ti = pg.TextItem(f"{conf:.2f}", anchor=(0.5, -1))
            font = QFont()
            font.setPointSize(14)    # larger size
            font.setBold(True)       # bold
            ti.setFont(font)
            self.xy_plot.addItem(ti)
            ti.setPos(c[1], c[0])
            self.text_items_xy.append(ti)

            ti2 = pg.TextItem(f"{conf:.2f}", anchor=(0.5, -1))
            ti2.setFont(font)
            self.yz_plot.addItem(ti2)
            ti2.setPos(c[2], c[1])
            self.text_items_yz.append(ti2)

        for c, conf in zip(non_centroids, non_confs):
            ti = pg.TextItem(f"{conf:.2f}", anchor=(0.5, -1))
            font = QFont()
            font.setPointSize(14)
            font.setBold(True)
            ti.setFont(font)
            self.xy_plot.addItem(ti)
            ti.setPos(c[1], c[0])
            self.text_items_xy.append(ti)

            ti2 = pg.TextItem(f"{conf:.2f}", anchor=(0.5, -1))
            ti2.setFont(font)
            self.yz_plot.addItem(ti2)
            ti2.setPos(c[2], c[1])
            self.text_items_yz.append(ti2)

        # distance & diagram
        d = self.distance_by_sec.get(sec)
        if d is None:
            self.dist_label.setText("Distance to dock: N/A")
            self.diagram_label.clear()
        else:
            self.dist_label.setText(f"Distance to dock: {d:.2f} m")
            self.draw_diagram(d)

    def run(self):
        self.win.show()
        self.app.exec_()


if __name__ == '__main__':
    CSV_PATH    = r"C:\Users\fotis\Downloads\Test\log_Tilos_12_04_30-07_21_labeled.csv"
    VIDEO_PATH  = r"C:\Users\fotis\Downloads\Tilos_12_04_30-07_21.mp4"
    MODEL_PATH  = r"C:\Users\fotis\Downloads\best_model.pt"
    SCALER_PATH = r"C:\Users\fotis\Downloads\gnn_scaler.pkl"

    viz = DockVisualizer(CSV_PATH, VIDEO_PATH, MODEL_PATH, SCALER_PATH)
    viz.run()



