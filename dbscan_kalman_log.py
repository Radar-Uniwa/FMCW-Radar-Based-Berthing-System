import os
import time
import pandas as pd
import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QGroupBox, QCheckBox, QHBoxLayout
from sklearn.cluster import DBSCAN

class RadarVisualization:
    def __init__(
        self,
        data_path,
        video_path,
        radar_columns=None,
        show_plot=True
    ):
        # 0) Load CSV & set defaults
        if radar_columns is None:
            radar_columns = ['AWR1443', 'AWR1642', 'AWR1843', 'IWR6843']
        self.data = pd.read_csv(data_path)
        self.video_path = video_path
        self.show_plot = show_plot

        # 1) Drop any radar columns not in CSV
        existing = set(self.data.columns)
        self.radar_columns = [c for c in radar_columns if c in existing]
        missing = set(radar_columns) - set(self.radar_columns)
        for col in missing:
            print(f"Warning: column '{col}' not found in CSV → dropping.")

        # 2) Read & filter the radar + time data
        self.radar_data, self.frame_ids, self.elapsed_time_formatted = self._prepare_data()

        # 3) If GUI mode, build the window & widgets
        if self.show_plot:
            self.app = QtWidgets.QApplication([])
            self.win = QtWidgets.QWidget()
            self.win.setWindowTitle("2D Radar Visualization + Logging")
            self._create_widgets()
        else:
            self.app = None
            self.win = None

        # 4) Create the animation/tracking controller
        self.controller = AnimationController(
            show_plot=self.show_plot,
            radar_data=self.radar_data,
            radar_columns=self.radar_columns,
            radar_checkboxes=(self.radar_checkboxes if self.show_plot else None),
            frame_ids=self.frame_ids,
            elapsed_time_formatted=self.elapsed_time_formatted,
            video_path=self.video_path,
            video_label=(self.video_label if self.show_plot else None),
            xy_plot_widget=(self.xy_plot_widget if self.show_plot else None),
            yz_plot_widget=(self.yz_plot_widget if self.show_plot else None),
            slider=(self.slider if self.show_plot else None),
            time_label=(self.time_label if self.show_plot else None),
        )

        # 5) Wire up GUI controls
        if self.show_plot:
            self.slider.valueChanged.connect(self.controller.on_slider_update)
            self.start_stop_button.clicked.connect(self.controller.on_start_stop)

    def _prepare_data(self):
        def filter_objects(obj_array, x_min=None, y_min=None, z_min=None,
                                        x_max=None, y_max=None, z_max=None):
            arr = np.array(obj_array, dtype=float)
            if arr.size == 0:
                return arr
            pts = np.atleast_2d(arr)
            if x_min is not None: pts = pts[pts[:,0] > x_min]
            if x_max is not None: pts = pts[pts[:,0] < x_max]
            if y_min is not None: pts = pts[pts[:,1] > y_min]
            if y_max is not None: pts = pts[pts[:,1] < y_max]
            if z_min is not None: pts = pts[pts[:,2] > z_min]
            if z_max is not None: pts = pts[pts[:,2] < z_max]
            return pts

        # Frame IDs
        if 'Frame_ID' in self.data.columns:
            frame_ids = self.data['Frame_ID'].to_numpy()
        else:
            frame_ids = np.arange(len(self.data))

        # Video metadata
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate   = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print("Video frames:", total_frames, "FPS:", frame_rate)

        # Keep only valid frames
        valid = frame_ids < total_frames
        frame_ids = frame_ids[valid]

        # Format elapsed times
        elapsed = [
            f"{int(f/frame_rate//60):02}:{int(f/frame_rate%60):02}"
            for f in frame_ids
        ]

        # Extract & filter each radar column
        radar_data = {}
        for col in self.radar_columns:
            raw = (
                self.data[col]
                .apply(lambda x: eval(x) if pd.notnull(x) else np.array([]))
                .to_numpy()[valid]
            )
            radar_data[col] = [filter_objects(arr) for arr in raw]

        return radar_data, frame_ids, elapsed

    def _create_widgets(self):
        layout = QtWidgets.QGridLayout()
        self.win.setLayout(layout)

        # Side view (X-Y)
        self.xy_plot_widget = pg.PlotWidget()
        self.xy_plot_widget.setBackground('w')
        self.xy_plot_widget.setLabel('left','X (m)')
        self.xy_plot_widget.setLabel('bottom','Y (m)')
        self.xy_plot_widget.setTitle("Side View (X-Y)")
        self.xy_plot_widget.showGrid(x=True,y=True,alpha=0.9)
        self.xy_plot_widget.setXRange(0,60)
        self.xy_plot_widget.setYRange(-15,15)
        layout.addWidget(self.xy_plot_widget,0,0)

        # Top view (Y-Z)
        self.yz_plot_widget = pg.PlotWidget()
        self.yz_plot_widget.setBackground('w')
        self.yz_plot_widget.setLabel('left','Y (m)')
        self.yz_plot_widget.setLabel('bottom','Z (m)')
        self.yz_plot_widget.setTitle("Top View (Y-Z)")
        self.yz_plot_widget.showGrid(x=True,y=True,alpha=0.9)
        self.yz_plot_widget.setXRange(-15,15)
        self.yz_plot_widget.setYRange(0,60)
        layout.addWidget(self.yz_plot_widget,0,1)

        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(240,120)
        layout.addWidget(self.video_label,1,0,1,2)

        # Controls: time, slider, start/stop (row 2)
        ctrl = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel("Time: 00:00")
        ctrl.addWidget(self.time_label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.frame_ids)-1)
        self.slider.setValue(0)
        ctrl.addWidget(self.slider)
        self.start_stop_button = QtWidgets.QPushButton("Start/Stop")
        ctrl.addWidget(self.start_stop_button)
        layout.addLayout(ctrl,2,0,1,2)

        # Radar‐column checkboxes (row 3)
        gb = QGroupBox("Select Radars")
        hb = QHBoxLayout()
        self.radar_checkboxes = {}
        for col in self.radar_columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.toggled.connect(self.on_radar_selection_changed)
            hb.addWidget(cb)
            self.radar_checkboxes[col] = cb
        gb.setLayout(hb)
        layout.addWidget(gb,3,0,1,2)

    def on_radar_selection_changed(self, checked):
        if hasattr(self, 'controller') and self.controller:
            self.controller.clear_buffer()

    def run(self):
        if self.show_plot:
            self.win.show()
            self.app.exec_()
        else:
            while self.controller.current_frame < len(self.frame_ids):
                self.controller.update()
                time.sleep(0.001)
            self.controller.process_final_buffer()

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        out = f"log_{base}.csv"
        self.controller.write_log_to_csv(out)
        print("Wrote log to", out)


class AnimationController:
    def __init__(
        self,
        show_plot,
        radar_data,
        radar_columns,
        radar_checkboxes,
        frame_ids,
        elapsed_time_formatted,
        video_path,
        video_label=None,
        xy_plot_widget=None,
        yz_plot_widget=None,
        slider=None,
        time_label=None
    ):
        self.show_plot   = show_plot
        self.radar_data  = radar_data
        self.radar_columns = radar_columns
        self.radar_checkboxes = radar_checkboxes
        self.frame_ids   = frame_ids
        self.elapsed_time_formatted = elapsed_time_formatted
        self.video_path  = video_path
        self.video_label = video_label
        self.xy_plot_widget = xy_plot_widget
        self.yz_plot_widget = yz_plot_widget
        self.slider      = slider
        self.time_label  = time_label

        # playback/buffering
        self.current_frame = 0
        self.cap = cv2.VideoCapture(video_path)
        self.buffer = []
        self.buffer_limit = 20
        self.buffer_start_frame = None

        # log
        self.log_entries = []

        # tracker
        self.tracker = MultiObjectTracker()

        if self.show_plot:
            # scatter items
            self.xy_detected  = pg.ScatterPlotItem(symbol='o', size=5, brush='r')
            self.yz_detected  = pg.ScatterPlotItem(symbol='o', size=5, brush='r')
            self.xy_centroids = pg.ScatterPlotItem(symbol='o', size=8, brush='b')
            self.yz_centroids = pg.ScatterPlotItem(symbol='o', size=8, brush='b')
            self.xy_predicted = pg.ScatterPlotItem(symbol='o', size=8, brush='y')
            self.yz_predicted = pg.ScatterPlotItem(symbol='o', size=8, brush='y')
            for item in (self.xy_detected, self.xy_centroids, self.xy_predicted):
                self.xy_plot_widget.addItem(item)
            for item in (self.yz_detected, self.yz_centroids, self.yz_predicted):
                self.yz_plot_widget.addItem(item)
            self.xy_boxes = []
            self.yz_boxes = []
            self.tracked_texts = []
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.is_running = False

    def update(self):
        # play video frame if GUI
        if self.show_plot and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                img = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(img).scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.video_label.setPixmap(pix)

        if not self.buffer:
            self.buffer_start_frame = self.current_frame

        # gather detections
        arrays = []
        if self.show_plot:
            for col, cb in self.radar_checkboxes.items():
                if cb.isChecked():
                    arr = self.radar_data[col][self.current_frame]
                    if arr.size > 0:
                        arrays.append(arr)
        else:
            for col in self.radar_columns:
                arr = self.radar_data[col][self.current_frame]
                if arr.size > 0:
                    arrays.append(arr)

        obj = np.vstack(arrays) if arrays else np.array([])
        if self.show_plot and obj.size > 0:
            self.xy_detected.setData(x=obj[:,1], y=obj[:,0])
            self.yz_detected.setData(x=obj[:,2], y=obj[:,1])

        self.buffer.append(obj)

        # process buffer
        if len(self.buffer) == self.buffer_limit:
            non_empty = [b for b in self.buffer if b.size > 0]
            if non_empty:
                comb = np.vstack(non_empty)
                centroids, clusters_info = self._update_clusters_and_centroids(comb)

                # 1) update tracker & get assigned IDs
                tids = self.tracker.update_tracks(centroids, self.current_frame)

                # 2) for GUI, still fetch predictions to visualize
                preds = self.tracker.get_tracked_objects()
                if self.show_plot:
                    self._update_visualization(centroids, preds)

                # 3) log one row *per cluster*
                self._log_buffer(clusters_info, tids)

            self.buffer = []

        # advance frame pointers
        if self.show_plot:
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
            self.current_frame = (self.current_frame + 1) % len(self.frame_ids)
            t = self.elapsed_time_formatted[self.current_frame]
            self.time_label.setText(f"Time: {t} | Frame: {self.current_frame}")
        else:
            self.current_frame += 1

    def process_final_buffer(self):
        if self.buffer:
            non_empty = [b for b in self.buffer if b.size > 0]
            if non_empty:
                comb = np.vstack(non_empty)
                centroids, clusters_info = self._update_clusters_and_centroids(comb)

                tids = self.tracker.update_tracks(centroids, self.current_frame)
                self._log_buffer(clusters_info, tids)

            self.buffer = []

    def _update_clusters_and_centroids(self, pts):
        db = DBSCAN(eps=3, min_samples=20).fit(pts[:,:3])
        labels = db.labels_
        centroids = []
        clusters_info = []
        new_xy_boxes = []
        new_yz_boxes = []

        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            cluster_pts = pts[labels == lbl]
            cen = cluster_pts[:,:3].mean(axis=0)
            centroids.append(cen)

            # bounding boxes (unused for logging, but kept for GUI)
            mx, Mx = cluster_pts[:,0].min(), cluster_pts[:,0].max()
            my, My = cluster_pts[:,1].min(), cluster_pts[:,1].max()
            mz, Mz = cluster_pts[:,2].min(), cluster_pts[:,2].max()

            if self.show_plot:
                bx = pg.RectROI([my, mx], [My-my, Mx-mx], pen=pg.mkPen('g',width=2))
                by = pg.RectROI([mz, my], [Mz-mz, My-my], pen=pg.mkPen('b',width=2))
                new_xy_boxes.append(bx)
                new_yz_boxes.append(by)

            # <-- full 5-dim point list here -->
            points_list = [
                [round(v,2) for v in pt]   # pt = [x,y,z,doppler,snr]
                for pt in cluster_pts[:, :5]
            ]

            clusters_info.append({
                "dbscan_cluster_id": int(lbl),
                "centroid":         [float(round(v,2)) for v in cen],
                "cluster_size":     int(len(cluster_pts)),
                "bounding_box_xy":  [round(mx,2),round(Mx,2),round(my,2),round(My,2)],
                "bounding_box_yz":  [round(my,2),round(My,2),round(mz,2),round(Mz,2)],
                "points":           points_list
            })

        if self.show_plot:
            for r in self.xy_boxes: self.xy_plot_widget.getViewBox().removeItem(r)
            for r in self.yz_boxes: self.yz_plot_widget.getViewBox().removeItem(r)
            self.xy_boxes, self.yz_boxes = new_xy_boxes, new_yz_boxes
            for r in self.xy_boxes: self.xy_plot_widget.getViewBox().addItem(r)
            for r in self.yz_boxes: self.yz_plot_widget.getViewBox().addItem(r)

        return np.array(centroids), clusters_info

    def _update_visualization(self, centroids, preds):
        # (unchanged: draws centroids + predicted tracks)
        self.xy_predicted.clear()
        self.yz_predicted.clear()
        self.xy_centroids.clear()
        self.yz_centroids.clear()
        for txt in self.tracked_texts:
            self.xy_plot_widget.removeItem(txt)
            self.yz_plot_widget.removeItem(txt)
        self.tracked_texts = []

        for t in getattr(self, 'tracked_texts', []):
            self.xy_plot_widget.removeItem(t)
            self.yz_plot_widget.removeItem(t)
        self.tracked_texts = []

        if centroids.size > 0:
            self.xy_centroids.setData(x=centroids[:,1], y=centroids[:,0])
            self.yz_centroids.setData(x=centroids[:,2], y=centroids[:,1])

        if preds:
            arr_xy = np.array([[p[1][1],p[1][0]] for p in preds])
            arr_yz = np.array([[p[1][2],p[1][1]] for p in preds])
            self.xy_predicted.setData(x=arr_xy[:,0], y=arr_xy[:,1])
            self.yz_predicted.setData(x=arr_yz[:,0], y=arr_yz[:,1])

            fnt = QFont(); fnt.setPointSize(14)
            for tid, pos in preds:
                t1 = pg.TextItem(f"ID {tid}", color="b", anchor=(0.5,0))
                t1.setFont(fnt); t1.setPos(pos[1],pos[0])
                self.xy_plot_widget.addItem(t1); self.tracked_texts.append(t1)
                t2 = pg.TextItem(f"ID {tid}", color="b", anchor=(0.5,0))
                t2.setFont(fnt); t2.setPos(pos[2],pos[1])
                self.yz_plot_widget.addItem(t2); self.tracked_texts.append(t2)

    def _log_buffer(self, clusters_info, tids):
        """
        Write one log row per cluster; 'tids' is the
        list of tracking IDs aligned with clusters_info.
        """
        # the frame at which this buffer ended
        end_f = min(self.current_frame, len(self.frame_ids) - 1)
        t1 = self.elapsed_time_formatted[end_f]
        frame_id = self.frame_ids[end_f]

        for ci, tid in zip(clusters_info, tids):
            self.log_entries.append({
                "Time":         t1,
                "Frame_ID":     int(frame_id),
                "Tracking_ID":  int(tid),
                "X":            ci["centroid"][0],
                "Y":            ci["centroid"][1],
                "Z":            ci["centroid"][2],
                "Num_Points":   ci["cluster_size"],
                "Points":       ci["points"],   # full [x,y,z,doppler,snr]
            })

        self.buffer_start_frame = None

    def on_slider_update(self, val):
        self.current_frame = val
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    def on_start_stop(self):
        if self.is_running:
            self.timer.stop()
        else:
            self.timer.start(50)
        self.is_running = not self.is_running

    def clear_buffer(self):
        self.buffer = []

    def write_log_to_csv(self, filename):
        # build DataFrame
        df = pd.DataFrame(self.log_entries)

        # sort by the frame index (ascending = earliest first)
        df = df.sort_values(by=['Frame_ID', 'Tracking_ID'])

        # write out
        df.to_csv(filename, index=False)
        print(f"Log saved → {filename}")



class KalmanFilter3D:
    def __init__(self):
        self.dt = 1
        self.F = np.array([
            [1,0,0,self.dt,0,0],
            [0,1,0,0,self.dt,0],
            [0,0,1,0,0,self.dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])
        self.H = np.eye(3,6)
        self.P = np.eye(6)*1000
        self.Q = np.eye(6)*0.1
        self.R = np.eye(3)*5

    def initialize(self, meas):
        self.state = np.zeros((6,1))
        self.state[:3,0] = meas

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.state[:3].flatten()

    def update(self, meas):
        z = np.array(meas).reshape(3,1)
        y = z - self.H.dot(self.state)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.state += K.dot(y)
        self.P = (np.eye(6)-K.dot(self.H)).dot(self.P)


class MultiObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.max_missing_frames = 20

    def update_tracks(self, detected_objects, frame_id):
        assignments = []
        updated     = {}
        objs        = np.atleast_2d(detected_objects) if detected_objects.size else []

        # Make a mutable copy of the pool
        free_trackers = dict(self.trackers)

        for obj in objs:
            pos     = obj[:3]
            matched = False

            # Only consider trackers still in free_trackers
            for tid, (kf, last_seen) in list(free_trackers.items()):
                pred = kf.predict()
                if np.linalg.norm(pred - pos) < 5:
                    # good match → update that filter & lock it
                    kf.update(pos)
                    updated[tid] = (kf, frame_id)
                    assignments.append(tid)
                    del free_trackers[tid]   # no longer available this cycle
                    matched = True
                    break

            if not matched:
                # brand-new object → new track ID
                new_id = self.next_id
                kf     = KalmanFilter3D()
                kf.initialize(pos)
                updated[new_id] = (kf, frame_id)
                assignments.append(new_id)
                self.next_id += 1

        # prune stale trackers as before
        self.trackers = {
            tid:(kf, ls)
            for tid, (kf, ls) in updated.items()
            if frame_id - ls < self.max_missing_frames
        }

        return assignments


    def get_tracked_objects(self):
        # For visualization: (tid, predicted_position)
        return [(tid, kf.predict()) for tid, (kf, _) in self.trackers.items()]


if __name__ == '__main__':
    ans = input("Show plot? (y/n): ").strip().lower()
    show = ans.startswith('y')

    all_radars = ['AWR1443', 'AWR1642', 'AWR1843', 'IWR6843']
    if not show:
        print("Select which radars to include (comma-separated):")
        for i, name in enumerate(all_radars, start=1):
            print(f"  {i}. {name}")
        print("  5. All of the above")
        sel = input("Your choice(s): ").strip()
        choices = {s.strip() for s in sel.split(',')}
        if '5' in choices:
            radars = all_radars
        else:
            radars = [
                all_radars[int(c)-1]
                for c in choices
                if c.isdigit() and 1 <= int(c) <= 4
            ]
        if not radars:
            print("No valid selection made; defaulting to all.")
            radars = all_radars
    else:
        radars = None

    DATA = r"path_to_synchronized_data.csv"
    VID  = r"path_to_video.mp4"

    viz = RadarVisualization(
        data_path=DATA,
        video_path=VID,
        radar_columns=radars,
        show_plot=show
    )
    viz.run()
