import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import os

def main():
    # 1) Load CSV
    path = input("Enter path to combined_labeled_trimmed.csv: ").strip().strip('"')
    if not os.path.exists(path):
        print("❌ File not found:", path)
        return

    df = pd.read_csv(path, encoding="utf-8-sig")

    # 2) Clean up Label column
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df = df[df["Label"] != "nan"]  # drop any noise-encoded rows

    print("\nClass distribution:")
    print(df["Label"].value_counts(normalize=True))

    # 3) Separate features & target
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # 4) Drop any raw/unneeded cols
    X = X.drop(
        columns=[
            "Frame", "Frame_ID",
            "Kalman_ID", "Tracking_ID",
            "Points", "Buffer_End_Frame"
        ],
        errors='ignore'
    )

    # 5) Rename delta columns for clarity
    X = X.rename(columns={
        "ΔX": "Delta_X",
        "ΔY": "Delta_Y",
        "ΔZ": "Delta_Z"
    })

    # 6) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 7) Split: train+val vs test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y_enc,
        test_size=0.15,
        random_state=42,
        stratify=y_enc
    )
    #    then train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=0.1765,  # so that val ≈15% of full
        random_state=42,
        stratify=y_tmp
    )

    print(f"\nDataset sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # 8) Train RF
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 9) Evaluate on validation
    y_val_pred = clf.predict(X_val)
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    # 10) Evaluate on test
    y_test_pred = clf.predict(X_test)
    print(f"\nTest Accuracy:       {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # 11) Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 12) Feature importances
    imps = clf.feature_importances_
    names = X.columns
    idxs = imps.argsort()[::-1]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(names)), imps[idxs], align='center')
    plt.xticks(range(len(names)), names[idxs], rotation=45, ha='right')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

    print("\nSorted feature importances:")
    for i in idxs:
        print(f"  {names[i]:30s}: {imps[i]:.6f}")

    # 13) Save model + encoder
    outdir = os.path.dirname(path)
    save_path = os.path.join(outdir, "rf_model_with_val.joblib")
    joblib.dump((clf, le), save_path)
    print(f"\n💾 Saved classifier + label encoder to {save_path}")

if __name__ == "__main__":
    main()
