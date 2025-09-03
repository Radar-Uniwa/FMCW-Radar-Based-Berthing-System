import os
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.stats import entropy

def extract_features(df):
    feature_rows = []
    prev_by_id = {}

    for _, row in df.sort_values(['Tracking_ID','Frame_ID']).iterrows():
        kid, frame = row['Tracking_ID'], row['Frame_ID']
        X, Y, Z     = row['X'], row['Y'], row['Z']

        # parse points into an (N×5) array
        pts = np.array(row['Points'], dtype=float)
        if pts.size == 0:
            continue

        coords, dop, snr = pts[:,:3], pts[:,3], pts[:,4]

        # normalized SNR
        norm_snr = snr + 20.0*np.log10(np.abs(coords[:,1]) + 1e-6)

        # doppler / snr statistics
        mean_d, std_d = dop.mean(), dop.std()
        mean_s, std_s = snr.mean(), snr.std()
        mean_ns, std_ns = norm_snr.mean(), norm_snr.std()
        max_d, range_d = dop.max(), np.ptp(dop)
        max_s, range_s = snr.max(), np.ptp(snr)
        ent_d = entropy(np.histogram(dop, bins=10, density=True)[0] + 1e-6)
        ent_s = entropy(np.histogram(snr, bins=10, density=True)[0] + 1e-6)
        corr_ds = np.corrcoef(snr, dop)[0,1] if len(snr) > 1 else 0.0

        # PCA‐based geometry features
        centered = coords - coords.mean(axis=0)
        cov      = np.cov(centered.T)
        eigs     = np.linalg.eigvalsh(cov)[::-1]
        λ1, λ2, λ3 = eigs
        linearity  = (λ1 - λ2) / (λ1 + 1e-6)
        planarity  = (λ2 - λ3) / (λ1 + 1e-6)
        sphericity =  λ3 / (λ1 + 1e-6)
        elongation = 1 - λ2 / (λ1 + 1e-6)
        eig_r12    = λ1 / (λ2 + 1e-6)
        eig_r23    = λ2 / (λ3 + 1e-6)

        # distances and compactness
        N     = len(pts)
        dists = np.linalg.norm(centered, axis=1)
        compactness   = dists.sum() / N
        avg_dist      = dists.mean()

        # bounding‐box, volume, density (fixed ptp usage)
        bbox    = np.ptp(coords, axis=0)
        volume  = np.prod(bbox + 1e-6)
        density = N / (volume + 1e-6)

        # assemble base features
        f = {
            'Frame_ID': frame, 'Tracking_ID': kid,
            'X': X, 'Y': Y, 'Z': Z,
            'Num_Points': N,
            'mean_doppler': mean_d,   'std_doppler': std_d,
            'mean_snr': mean_s,       'std_snr': std_s,
            'norm_mean_snr': mean_ns, 'norm_std_snr': std_ns,
            'doppler_max': max_d,     'doppler_range': range_d,
            'snr_max': max_s,         'snr_range': range_s,
            'doppler_entropy': ent_d, 'snr_entropy': ent_s,
            'snr_doppler_corr': corr_ds,
            'eigen_1': λ1, 'eigen_2': λ2, 'eigen_3': λ3,
            'linearity': linearity, 'planarity': planarity,
            'sphericity': sphericity, 'elongation': elongation,
            'eig_r12': eig_r12, 'eig_r23': eig_r23,
            'compactness': compactness, 'volume': volume,
            'density': density, 'avg_dist_to_centroid': avg_dist,
            'Dock_Label': row['Dock_Label']
        }

        # temporal deltas
        prev = prev_by_id.get(kid)
        if prev:
            f['ΔX'] = X - prev['X']
            f['ΔY'] = Y - prev['Y']
            f['ΔZ'] = Z - prev['Z']
            f['speed'] = np.linalg.norm([f['ΔX'], f['ΔY'], f['ΔZ']])
            f['velocity_angle'] = np.arctan2(f['ΔY'], f['ΔX']) if f['ΔX'] != 0 else 0.0

            for key in list(f):
                if key not in ('Frame_ID','Tracking_ID','Dock_Label','ΔX','ΔY','ΔZ','speed','velocity_angle'):
                    f['Δ'+key] = f[key] - prev.get(key, 0.0)
        else:
            # initialize deltas to zero on first appearance
            for key in list(f):
                if key not in ('Frame_ID','Tracking_ID','Dock_Label'):
                    f['Δ'+key] = 0.0

        prev_by_id[kid] = f
        feature_rows.append(f)

    return pd.DataFrame(feature_rows)


def process_all_csvs(input_folder):
    # Create output subfolder
    out_folder = os.path.join(input_folder, "features")
    os.makedirs(out_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith('.csv') or fname.lower().endswith('_features.csv'):
            continue
        in_path  = os.path.join(input_folder, fname)
        out_name = fname.replace('.csv','_features.csv')
        out_path = os.path.join(out_folder, out_name)

        # Load & parse
        df = pd.read_csv(in_path, encoding='utf-8-sig')
        df['Points'] = df['Points'].apply(lambda s: literal_eval(s) if pd.notna(s) else [])
        df = df[df['Points'].apply(len) > 0]

        # Extract & save
        feat_df = extract_features(df)
        feat_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"✅ {fname} → features/{out_name}")


if __name__ == "__main__":
    folder = input("Enter folder path containing your CSVs: ").strip().strip('"')
    process_all_csvs(folder)
