import os
import pandas as pd
import ast
import re

# ------------------ Functions ------------------

def clean_and_parse_detected_objects(obj):
    """Parses a string of detected objects into a list of sublists,
    returning only sublists with at least 3 elements."""
    try:
        if not isinstance(obj, str):
            return []  # Ensure input is a string
        # Wrap content with brackets
        wrapped_obj = f"[{obj}]"  # Add outer brackets
        parsed = ast.literal_eval(wrapped_obj)
        if isinstance(parsed, list):
            return [sublist for sublist in parsed 
                    if isinstance(sublist, (list, tuple)) and len(sublist) >= 3]
        return []
    except Exception as e:
        print(f"Error parsing object: {obj[:100]}... - {e}")
        return []

def filter_points(points, thresholds):
    """Filters a list of points so that each point falls within the provided thresholds."""
    filtered = []
    for sublist in points:
        if (thresholds['x'][0] <= sublist[0] <= thresholds['x'][1] and
            thresholds['y'][0] <= sublist[1] <= thresholds['y'][1] and
            thresholds['z'][0] <= sublist[2] <= thresholds['z'][1]):
            filtered.append(sublist)
    return filtered

def get_threshold_input(axis, default_min, default_max):
    """Prompts for min and max thresholds for an axis (interactive)."""
    try:
        min_value = input(f"  Minimum {axis} (press Enter to use {default_min}): ")
        min_value = float(min_value) if min_value.strip() else default_min
    except ValueError:
        min_value = default_min
    try:
        max_value = input(f"  Maximum {axis} (press Enter to use {default_max}): ")
        max_value = float(max_value) if max_value.strip() else default_max
    except ValueError:
        max_value = default_max
    return min_value, max_value

def compute_global_thresholds(file_list, radar_columns):
    """Scans all files to extract valid points from the given radar columns,
    then computes global minimum and maximum values for x, y, and z."""
    all_points = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        for radar in radar_columns:
            if radar in df.columns:
                for obj in df[radar].dropna():
                    points = clean_and_parse_detected_objects(obj)
                    all_points.extend(points)
    if all_points:
        default_min = [min(point[i] for point in all_points) for i in range(3)]
        default_max = [max(point[i] for point in all_points) for i in range(3)]
    else:
        default_min = [float('-inf')] * 3
        default_max = [float('inf')] * 3
    return default_min, default_max

def process_file(file, radar_columns, global_thresholds):
    """Processes one CSV file: applies filtering on the specified radar columns and saves the result."""
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return
    for radar in radar_columns:
        if radar in df.columns:
            # Parse the string, filter points, and then join them back to a string.
            df[radar] = df[radar].apply(
                lambda obj: ', '.join(map(str, filter_points(clean_and_parse_detected_objects(obj), global_thresholds)))
            )
            # If no valid objects remain, leave the field empty.
            df[radar] = df[radar].apply(lambda x: x if x.strip() else '')
        else:
            print(f"  Column '{radar}' not found in {file}.")
    base_name, ext = os.path.splitext(file)
    output_file = f"{base_name}_filtered{ext}"
    try:
        df.to_csv(output_file, index=False)
        print(f"Filtered data saved to {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

# ------------------ Main Execution ------------------

# Ask for the base directory
base_dir = input("Enter the base directory (e.g., G:\\Dataset1): ").strip('"')
if not os.path.isdir(base_dir):
    print(f"Directory {base_dir} not found.")
    exit()

# Recursively find all files named "merged_detected_objects_by_radar.csv"
file_list = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "merged_detected_objects_by_radar.csv":
            file_list.append(os.path.join(root, file))

if not file_list:
    print("No merged_detected_objects_by_radar.csv files found.")
    exit()

print(f"Found {len(file_list)} file(s).")

# Radar columns to process
radar_columns = ['AWR1443', 'AWR1642', 'AWR1843', 'IWR6843']

# Compute global thresholds from all found files
global_default_min, global_default_max = compute_global_thresholds(file_list, radar_columns)
print("Global default thresholds for all radar columns:")
print(f"  X: {global_default_min[0]} to {global_default_max[0]}")
print(f"  Y: {global_default_min[1]} to {global_default_max[1]}")
print(f"  Z: {global_default_min[2]} to {global_default_max[2]}")

# Prompt the user for thresholds (these will be applied to all files)
global_thresholds = {
    'x': get_threshold_input('X', global_default_min[0], global_default_max[0]),
    'y': get_threshold_input('Y', global_default_min[1], global_default_max[1]),
    'z': get_threshold_input('Z', global_default_min[2], global_default_max[2]),
}

# Process each file individually
for file in file_list:
    print(f"\nProcessing file: {file}")
    process_file(file, radar_columns, global_thresholds)
