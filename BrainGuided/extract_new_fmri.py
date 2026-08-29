"""
Extract fMRI samples matching new video IDs
============================================
The new CSVs removed clip_000 videos and re-numbered all IDs from scratch.
The same video therefore has a DIFFERENT ID in old vs new.

Matching logic (per split):
  1. Build name → old_id  map from old CSV
  2. Build new_id → name  map from new CSV
  3. Combine: new_id → old_id  (via shared video name)
  4. New mat file gives new_ids in scan order
     → convert to old_ids → find rows in old fMRI → reorder
  5. Save new fMRI with rows in new-mat order

Output files (saved next to originals in each sub-{subject}/ folder):
  rt   train : voxel_select_new_ALL_{split}_new.npy          (720, V)
  rt   test  : voxel_select_new_valid_ALL_{split}_new.npy    (240, V)
  annot train: voxel_select_new_annot_ALL_{split}_new.npy    (360, V)
  annot test : voxel_select_new_annot_valid_ALL_{split}_new.npy (120, V)
"""

import os
import numpy as np
import pandas as pd
import h5py

# ──────────────────────────────────────────────
# Configuration  ← edit these paths
# ──────────────────────────────────────────────
FMRI_ROOT   = "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results"

# Old mat files (original video order)
MAT_DIR_OLD = "/mnt/d/IDWB/Video-Emotion/BrainGuided/old_files"
# New mat files (new video order, new IDs) — put them in a subfolder
MAT_DIR_NEW = "/mnt/d/IDWB/Video-Emotion/Neural_data/"

# Old CSVs
CSV_RT_OLD    = "/mnt/d/IDWB/Video-Emotion/BrainGuided/old_files/video_id_rt.csv"
CSV_ANNOT_OLD = "/mnt/d/IDWB/Video-Emotion/BrainGuided/old_files/video_id_rt_annot.csv"
# New CSVs
CSV_RT_NEW    = "/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv"
CSV_ANNOT_NEW = "/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt_annot.csv"

SUBJECTS = ["01", "02", "03", "06", "08", "09", "12"]
N_SPLITS = 4
 
# Each task defines one pair of output files (train + test) per split
# old_train_pat / old_test_pat: existing fMRI files to pool together
# old_train_mat / old_test_mat: old mat files to align the pool rows
# new_train_mat / new_test_mat: new mat files defining desired output order
TASKS = [
    dict(
        name          = "rt",
        old_train_pat = "voxel_select_new_ALL_{split}.npy",
        old_test_pat  = "voxel_select_new_valid_ALL_{split}.npy",
        old_train_mat = "video_order_rt_{split}.mat",
        old_test_mat  = "video_order_rt_valid_{split}.mat",
        new_train_mat = "video_order_rt_{split}.mat",
        new_test_mat  = "video_order_rt_valid_{split}.mat",
        out_train_pat = "voxel_select_new_ALL_{split}_new.npy",
        out_test_pat  = "voxel_select_new_valid_ALL_{split}_new.npy",
        csv_key       = "rt",
    ),
    dict(
        name          = "annot",
        old_train_pat = "voxel_select_new_annot_ALL_{split}.npy",
        old_test_pat  = "voxel_select_new_annot_valid_ALL_{split}.npy",
        old_train_mat = "video_order_rt_annot_{split}.mat",
        old_test_mat  = "video_order_rt_annot_valid_{split}.mat",
        new_train_mat = "video_order_rt_annot_{split}.mat",
        new_test_mat  = "video_order_rt_annot_valid_{split}.mat",
        out_train_pat = "voxel_select_new_annot_ALL_{split}_new.npy",
        out_test_pat  = "voxel_select_new_annot_valid_ALL_{split}_new.npy",
        csv_key       = "annot",
    ),
]
 
# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
 
def norm(s: str) -> str:
    return s.replace('.mp4', '').strip()
 
 
def build_name_maps(csv_old: str, csv_new: str):
    """
    Returns
    -------
    new_id_to_old_id : dict  {new_video_id -> old_video_id}  via shared video name
    """
    df_old = pd.read_csv(csv_old)
    df_new = pd.read_csv(csv_new)
    df_old['name_norm'] = df_old['Video Name and Directory'].apply(norm)
    df_new['name_norm'] = df_new['Video Name and Directory'].apply(norm)
 
    name_to_old_id = dict(zip(df_old['name_norm'], df_old['Video ID']))
    new_id_to_name = dict(zip(df_new['Video ID'],  df_new['name_norm']))
 
    mapping, missing = {}, []
    for new_id, name in new_id_to_name.items():
        old_id = name_to_old_id.get(name)
        if old_id is not None:
            mapping[new_id] = old_id
        else:
            missing.append((new_id, name))
    if missing:
        print(f"  [WARN] {len(missing)} video names in new CSV not found in old CSV "
              f"(should be 0): {missing[:3]}")
    return mapping
 
 
def load_mat(mat_dir: str, filename: str) -> np.ndarray:
    with h5py.File(os.path.join(mat_dir, filename), 'r') as f:
        return np.array(f['video_order']).flatten().astype(int)
 
 
def build_pool(subj_dir, train_pat, test_pat, train_mat, test_mat, split):
    """
    Concatenate old train + test fMRI for this split into one array and
    build a dict: old_video_id → row_index_in_pool.
    """
    tr_path = os.path.join(subj_dir, train_pat.format(split=split))
    te_path = os.path.join(subj_dir,  test_pat.format(split=split))
 
    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"Missing: {tr_path}")
    if not os.path.exists(te_path):
        raise FileNotFoundError(f"Missing: {te_path}")
 
    fmri_tr = np.load(tr_path)   # (N_tr, V)
    fmri_te = np.load(te_path)   # (N_te, V)
    pool    = np.concatenate([fmri_tr, fmri_te], axis=0)   # (N_tr+N_te, V)
 
    old_tr_ids = load_mat(MAT_DIR_OLD, train_mat.format(split=split))
    old_te_ids = load_mat(MAT_DIR_OLD,  test_mat.format(split=split))
 
    id_to_row = {}
    for i, vid in enumerate(old_tr_ids):
        id_to_row[vid] = i
    offset = len(old_tr_ids)
    for i, vid in enumerate(old_te_ids):
        id_to_row[vid] = offset + i
 
    return pool, id_to_row
 
 
def select_rows(pool, id_to_row, new_id_to_old_id, new_ids, label):
    """Select rows from pool in the order given by new_ids."""
    rows, skipped = [], []
    for new_id in new_ids:
        old_id = new_id_to_old_id.get(new_id)
        if old_id is None:
            skipped.append(('no_name_map', new_id))
            continue
        row = id_to_row.get(old_id)
        if row is None:
            skipped.append(('not_in_pool', new_id, old_id))
            continue
        rows.append(row)
 
    if skipped:
        print(f"    [WARN] {label}: {len(skipped)} IDs still missing after pooling: "
              f"{skipped[:3]}")
    return pool[np.array(rows)]
 
 
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
 
def main():
    print("Building video ID maps (old ↔ new via video name) ...")
    id_maps = {
        "rt"   : build_name_maps(CSV_RT_OLD,    CSV_RT_NEW),
        "annot": build_name_maps(CSV_ANNOT_OLD, CSV_ANNOT_NEW),
    }
    print(f"  rt    map: {len(id_maps['rt'])} videos")
    print(f"  annot map: {len(id_maps['annot'])} videos")
 
    for subject in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"  Subject: {subject}")
        print(f"{'='*60}")
        subj_dir = os.path.join(FMRI_ROOT, f"sub-{subject}")
 
        for split in range(1, N_SPLITS + 1):
            print(f"\n  Split {split}")
 
            for task in TASKS:
                name = task['name']
                try:
                    # Build unified pool from old train + test fMRI
                    pool, id_to_row = build_pool(
                        subj_dir,
                        task['old_train_pat'], task['old_test_pat'],
                        task['old_train_mat'], task['old_test_mat'],
                        split,
                    )
                except FileNotFoundError as e:
                    print(f"    [SKIP] {e}")
                    continue
 
                mapping = id_maps[task['csv_key']]
 
                # New train
                new_tr_ids = load_mat(MAT_DIR_NEW, task['new_train_mat'].format(split=split))
                new_tr_fmri = select_rows(pool, id_to_row, mapping, new_tr_ids,
                                          f"{name}_train")
                out_tr = os.path.join(subj_dir, task['out_train_pat'].format(split=split))
                np.save(out_tr, new_tr_fmri)
 
                # New test
                new_te_ids = load_mat(MAT_DIR_NEW, task['new_test_mat'].format(split=split))
                new_te_fmri = select_rows(pool, id_to_row, mapping, new_te_ids,
                                          f"{name}_test")
                out_te = os.path.join(subj_dir, task['out_test_pat'].format(split=split))
                np.save(out_te, new_te_fmri)
 
                print(f"    [{name}]  pool={pool.shape}  "
                      f"→ train={new_tr_fmri.shape}  test={new_te_fmri.shape}")
 
    print("\nDone. All '_new' fMRI files saved.")
 
 
if __name__ == "__main__":
    main()
 