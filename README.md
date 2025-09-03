# FMCW-Radar-Based-Berthing-System
This repository contains the Python code developed for the Radar-Based Berthing Aid System project. The project took place in University of West Attica, Greece, under the Master by Research. It includes preprocessing, feature engineering, training, and evaluation scripts for machine learning and deep learning models used to classify radar point clouds into Dock and No Dock categories.
The radars that were used were the AWR1443BOOST, AWR1642BOOST, AWR1843BOOST and IWR6843ISK from Texas Instruments. The implementation included a Raspberry Pi 4 and a supportive subsystem for measurements at the field, a Ro/Ro Pax ferry which was operating normaly. 
More about our system can be found here https://ieeexplore.ieee.org/document/11083964. 

## 📂 Repository Structure

```FMCW-Radar-Based-Berthing-System/
│
├── Preprocessing/               # Scripts for preparing raw radar data
│   ├── annotation.py             # Annotation tool for labeling clusters
│   ├── bin2csv_synchronized.py   # Convert binary radar data to synchronized CSV
│   ├── dbscan_kalman_log.py      # Run DBSCAN + Kalman and log results
│   ├── filter_csv.py             # Filter detections based on distance thresholds
│   ├── preprocess_feature.py     # Feature-based preprocessing
│   └── preprocess_gnn.py         # GNN-specific preprocessing
│
├── Training/                    # Model training scripts
│   ├── train_gnn.py              # Train Graph Neural Network
│   ├── train_pointnet.py         # Train PointNet model
│   ├── train_xgboost.py          # Train XGBoost classifier
│   └── training_with_features.py # Train Random Forest and other feature-based models
│
├── Feature_Importance/          # Feature analysis
│   ├── feature_importance_xgboost.py # Feature importance via XGBoost
│   └── features_importance_rf.py     # Feature importance via Random Forest
│
├── Extras/                      # Additional scripts / exploratory analysis
│   ├── pca_analysis.py           # PCA analysis 
│   └── visualization_gnn.py      # Visualization using GNN model for classifying clusters 
│
├── LICENSE                      # License file
└── README.md                    # Project documentation

```
