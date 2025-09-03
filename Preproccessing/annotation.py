import pandas as pd
import os

# 1) Provide file path
file_path = input("Enter the path to the log CSV file: ").strip()

# 2) Load CSV and strip whitespace from columns
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# 3) Verify it has the right columns
required = {"Tracking_ID", "X", "Y", "Z", "Frame_ID"}
if not required.issubset(df.columns):
    print("Missing required columns. Found:", df.columns.tolist())
    exit()

# 4) Show the IDs present
print("Available Tracking IDs:", sorted(df["Tracking_ID"].unique()))

# 5) Ask which of those IDs correspond to the dock
dock_ids_input = input(
    "Enter the Tracking IDs to label as dock (comma-separated): "
)
dock_ids = {
    int(x.strip())
    for x in dock_ids_input.split(",")
    if x.strip().isdigit()
}

# 6) Add the label-only column with 1/0
df["Dock_Label"] = df["Tracking_ID"].apply(
    lambda tid: 1 if tid in dock_ids else 0
)

# 7) Save back out
base, ext = os.path.splitext(file_path)
out = f"{base}_labeled{ext}"
df.to_csv(out, index=False)

print(f"\n✅ File saved to: {out}")
