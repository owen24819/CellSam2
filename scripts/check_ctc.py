import os
import re
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from tifffile import imread


def load_man_track_txt(txt_path):
    """
    Loads the manual tracking file (man_track.txt) in CTC format.
    Format: L B E P where:
        L - label (unique label of the track)
        B - begin frame (zero-based temporal index)
        E - end frame (zero-based temporal index)
        P - parent label (0 when no parent)
    
    Returns a dict: key: frame_idx (int), value: set of labels present in that frame
    """
    frame_to_labels = {}
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"{txt_path} not found")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if len(stripped) == 0:
                continue
            items = stripped.split()
            if len(items) < 4:
                continue
            # Parse L B E P format
            label, begin_frame, end_frame, _ = map(int, items[:4])  # parent not needed for this check
            # Add this label to all frames from begin_frame to end_frame (inclusive)
            for frame_idx in range(begin_frame, end_frame + 1):
                if frame_idx not in frame_to_labels:
                    frame_to_labels[frame_idx] = set()
                frame_to_labels[frame_idx].add(label)
    
    return frame_to_labels

def check_man_track_vs_tif(traj_folder):
    """
    Checks that every cell label in each man_trackXXX.tif frame is present in the man_track.txt for that frame, and vice versa.
    
    Returns:
        tuple: (errors_found: bool, error_count: int)
    """
    txt_path = os.path.join(traj_folder, "man_track.txt")
    if not os.path.exists(txt_path):
        print(f"  TRA folder exists but man_track.txt not found: {traj_folder}")
        return True, 1
    
    try:
        frame_to_labels = load_man_track_txt(txt_path)
    except Exception as e:
        print(f"  Error loading man_track.txt: {e}")
        return True, 1
    
    # Gather all tif files in this directory matching man_trackXXX.tif
    tifs = []
    if not os.path.exists(traj_folder):
        print(f"  TRA folder not found: {traj_folder}")
        return True, 1
    
    try:
        for fn in os.listdir(traj_folder):
            if fn.startswith("man_track") and fn.endswith(".tif") and fn != "man_track.txt":
                tifs.append(fn)
    except Exception as e:
        print(f"  Error listing files in TRA folder: {e}")
        return True, 1
    
    if len(tifs) == 0:
        print(f"  No man_trackXXX.tif files found in {traj_folder}")
        return True, 1
    
    errors_found = False
    error_count = 0
    
    for tif_fn in sorted(tifs):
        # Extract the frame idx from filename man_trackXXX.tif (XXX is a zero-padded int)
        fname = os.path.splitext(tif_fn)[0]
        frame_part = fname.split("man_track")[-1]
        if not frame_part.isdigit():
            print(f"  Could not parse frame index from {tif_fn}")
            errors_found = True
            error_count += 1
            continue
        
        frame_idx = int(frame_part)
        tif_path = os.path.join(traj_folder, tif_fn)
        try:
            mask = imread(tif_path)  # shape: (H, W)
            mask_labels = set(np.unique(mask))
            mask_labels.discard(0)  # 0 is background/not a cell
        except Exception as e:
            print(f"  Error reading {tif_fn}: {e}")
            errors_found = True
            error_count += 1
            continue

        txt_labels = frame_to_labels.get(frame_idx, set())

        in_mask_not_txt = mask_labels - txt_labels
        in_txt_not_mask = txt_labels - mask_labels

        if in_mask_not_txt:
            print(f"  Frame {frame_idx:03d}: Labels in man_track{frame_idx:03d}.tif but not in man_track.txt: {sorted(in_mask_not_txt)}")
            errors_found = True
            error_count += 1
        if in_txt_not_mask:
            print(f"  Frame {frame_idx:03d}: Labels in man_track.txt but not in man_track{frame_idx:03d}.tif: {sorted(in_txt_not_mask)}")
            errors_found = True
            error_count += 1

    # Check for frames in txt that do not have a corresponding tif
    for frame_idx in frame_to_labels.keys():
        tif_name = f"man_track{frame_idx:03d}.tif"
        if tif_name not in tifs:
            print(f"  man_track.txt has entries for frame {frame_idx:03d} but no {tif_name} in TRA folder!")
            errors_found = True
            error_count += 1

    if not errors_found:
        print("  ✓ All checks passed")
    
    return errors_found, error_count

def check_dataset(dataset_folder):
    """
    Check all sequences in train/CTC, val/CTC, and test/CTC for a given dataset.
    
    Args:
        dataset_folder: Path to the dataset folder (e.g., "moma")
    """
    dataset_folder = os.path.abspath(dataset_folder)
    dataset_name = os.path.basename(dataset_folder)
    
    splits = ["train", "val", "test"]
    all_errors = False
    total_error_count = 0
    
    for split in splits:
        split_path = os.path.join(dataset_folder, split, "CTC")
        if not os.path.exists(split_path):
            print(f"\n[{split}] CTC folder not found: {split_path}")
            continue
        
        # Find all folders matching pattern: two digits followed by "_GT" (e.g., "01_GT", "02_GT")
        gt_folders = []
        try:
            for item in os.listdir(split_path):
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path) and re.match(r'^\d{2}_GT$', item):
                    gt_folders.append((item, item_path))
        except Exception as e:
            print(f"\n[{split}] Error listing folders in {split_path}: {e}")
            continue
        
        if len(gt_folders) == 0:
            print(f"\n[{split}] No folders matching pattern XX_GT found in {split_path}")
            continue
        
        gt_folders.sort()  # Sort by folder name
        print(f"\n=== Checking {split} split ({len(gt_folders)} sequences) ===")
        
        for gt_folder_name, gt_folder_path in gt_folders:
            sequence_name = gt_folder_name.replace("_GT", "")
            tra_folder = os.path.join(gt_folder_path, "TRA")
            
            if not os.path.exists(tra_folder):
                print(f"[{split}/{sequence_name}] TRA folder not found: {tra_folder}")
                all_errors = True
                total_error_count += 1
                continue
            
            print(f"[{split}/{sequence_name}] Checking TRA folder...")
            has_errors, error_count = check_man_track_vs_tif(tra_folder)
            if has_errors:
                all_errors = True
                total_error_count += error_count
    
    print("\n" + "=" * 70)
    if not all_errors:
        print(f"✓ All checks passed for dataset '{dataset_name}': man_track.txt and man_trackXXX.tif files are consistent.")
    else:
        print(f"✗ Found {total_error_count} error(s) in dataset '{dataset_name}'")
    
    return all_errors, total_error_count

def main():
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open folder selection dialog
    folder = filedialog.askdirectory(title="Select Dataset Folder (e.g., moma)")
    
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    
    print(f"Checking dataset folder: {folder}")
    print("=" * 70)
    
    errors_found, error_count = check_dataset(folder)
    
    if errors_found:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
