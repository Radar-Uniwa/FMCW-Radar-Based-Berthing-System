import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IGNORE_COLS = ["Frame_ID", "Tracking_ID", "Dock_Label", "X", "Y", "Z"]

def feature_names_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df.drop(columns=IGNORE_COLS, errors="ignore").columns.tolist()

def xgb_importances_percent(clf, feature_names):
    imp = getattr(clf, "feature_importances_", None)

    if isinstance(imp, np.ndarray) and imp.size == len(feature_names):
        scores = imp.astype(float)
    else:
        # fallback: booster.get_score mapping
        booster = clf.get_booster()
        raw = booster.get_score(importance_type="gain")  # dict
        scores = np.zeros(len(feature_names), dtype=float)
        for k, v in raw.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    scores[idx] = float(v)
            elif k in feature_names:
                scores[feature_names.index(k)] = float(v)

    total = scores.sum()
    if total > 0:
        scores = 100.0 * scores / total
    return scores

def paired_barh(features, vals1, vals2, title, outfile=None):
    # order by average importance (descending)
    avg = (vals1 + vals2) / 2.0
    order = np.argsort(avg)[::-1][:20]   # top-20
    names  = [features[i] for i in order][::-1]   # reverse for ascending y
    v1     = [vals1[i] for i in order][::-1]
    v2     = [vals2[i] for i in order][::-1]

    y = np.arange(len(names))
    h = 0.38
    plt.figure(figsize=(11, max(5, 0.45*len(names))))
    plt.barh(y - h/2, v1, height=h, label="Split #1")
    plt.barh(y + h/2, v2, height=h, label="Split #2")

    plt.yticks(y, names)
    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
        print(f"✅ Saved plot to {outfile}")
    plt.show()

def main():
    m1 = input("Path to Split #1 XGB model (.pkl): ").strip().strip('"')
    c1 = input("Path to a Split #1 feature CSV: ").strip().strip('"')
    m2 = input("Path to Split #2 XGB model (.pkl): ").strip().strip('"')
    c2 = input("Path to a Split #2 feature CSV: ").strip().strip('"')

    clf1 = joblib.load(m1)
    clf2 = joblib.load(m2)
    fn1 = feature_names_from_csv(c1)
    fn2 = feature_names_from_csv(c2)

    # sanity: unify feature name order using Split #1 as reference
    if fn1 != fn2:
        # align Split #2 scores to Split #1 order (missing → 0)
        name_to_idx2 = {n:i for i,n in enumerate(fn2)}
        aligned2 = [name_to_idx2.get(n, None) for n in fn1]
    else:
        aligned2 = list(range(len(fn1)))

    scores1 = xgb_importances_percent(clf1, fn1)
    raw2    = xgb_importances_percent(clf2, fn2)
    scores2 = np.array([raw2[i] if i is not None else 0.0 for i in aligned2])

    paired_barh(
        fn1,
        scores1,
        scores2,
        title="Top 20 Feature Importances (XGBoost) – Split #1 vs Split #2",
        outfile="xgb_feature_importances_dual.png"
    )

if __name__ == "__main__":
    main()
