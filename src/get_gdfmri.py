#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI Feature Extraction Script

This script processes fMRI data from a GLMdenoise analysis. For each subject and
session, it performs the following steps:
1.  Loads single-trial beta-series data from all functional runs.
2.  Loads corresponding event files to identify conditions (trial_types).
3.  Averages the beta maps for each unique condition across all runs.
4.  Loads predefined Region of Interest (ROI) masks and a subject-specific SNR mask.
    The SNR mask is thresholded to include only voxels with values > 40.
5.  Applies the combined ROI and thresholded SNR masks to the averaged beta maps.
6.  Saves the resulting feature vector for each ROI (including SNR) and condition as a .npy file.

This script assumes that the 4th dimension of the input NIfTI files 
('...desc-denoised_bold.nii.gz') corresponds to the trials listed in the 
'events_aug.tsv' files.
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.image import resample_to_img

import os
import sys
import numpy as np
import nibabel as nib
from collections import defaultdict


def load_roi_masks(rois_base_path, roi_names):
    """
    Loads NIfTI images for specified ROIs. If 'ALL' is requested, it ensures
    the component ROIs (RSC, PPA, EVC, TOS) are loaded individually.
    """
    print(f"Loading ROI masks from: {rois_base_path}")
    roi_imgs = {}
    loaded_roi_cache = {}

    # <<< CHANGE: Determine the full set of ROIs to load
    rois_to_load_set = set(roi_names)
    if "ALL" in rois_to_load_set:
        rois_to_load_set.remove("ALL")
        rois_to_load_set.update(['RSC', 'PPA', 'EVC', 'TOS'])

    def _get_single_roi_img(name):
        # (This helper function remains the same as your original code)
        if name in loaded_roi_cache:
            return loaded_roi_cache[name]
        
        print(f"    - Loading component: {name}...")
        
        img = None
        if name == "EVC":
            path = os.path.join(rois_base_path, "V1.nii.gz")
            if not os.path.exists(path):
                print(f"      ERROR: EVC file not found at {path}")
                return None
            img = nib.load(path)
        else:
            l_path = os.path.join(rois_base_path, f'l{name}.img')
            r_path = os.path.join(rois_base_path, f'r{name}.img')
            if not os.path.exists(l_path) or not os.path.exists(r_path):
                print(f"      ERROR: Left or right hemisphere file not found for {name}")
                return None
            img_l = nib.load(l_path)
            img_r = nib.load(r_path)
            combined_data = np.logical_or(img_l.get_fdata(), img_r.get_fdata())
            img = nib.Nifti1Image(combined_data.astype(np.float32), affine=img_l.affine)
        
        loaded_roi_cache[name] = img
        return img

    try:
        # <<< CHANGE: Loop over the expanded set of ROIs
        for roi in rois_to_load_set:
            print(f"  - Loading individual ROI: {roi}...")
            img = _get_single_roi_img(roi)
            if img:
                roi_imgs[roi] = img

        print("ROI masks loaded successfully.")
        return roi_imgs
        
    except Exception as e:
        print(f"An error occurred while loading ROI masks: {e}", file=sys.stderr)
        return None


def process_subject_session(sub, ses, data_base_path, original_roi_niftis, requested_rois, output_base_path, tr_seconds, num_trs_to_average):
    """
    Processes all data for a single subject's session.
    """
    print("-" * 50)
    print(f"Processing sub-{sub}, ses-{ses}")

    # --- 1 & 2. Define paths and find run files (No changes here) ---
    glm_results_path = os.path.join(data_base_path, 'derivatives', 'glm_denoise_results', f'sub-{sub}', f'ses-{ses}')
    func_path = os.path.join(data_base_path, f'sub-{sub}', f'ses-{ses}', 'func')
    run_bold_files = sorted(glob.glob(os.path.join(glm_results_path, f'sub-{sub}_ses-{ses}_task-roomtour_run-*_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz')))
    run_event_files = sorted(glob.glob(os.path.join(func_path, f'sub-{sub}_ses-{ses}_task-roomtour_run-*_events_aug.tsv')))
    if not run_bold_files or not run_event_files:
        print(f"ERROR: No BOLD or event files found for sub-{sub}, ses-{ses}. Skipping.", file=sys.stderr)
        return

    # --- 3. Prepare ROI masks ---
    first_bold_nii = nib.load(run_bold_files[0])
    volume_shape = first_bold_nii.shape[:3]
    target_img_for_resampling = nib.Nifti1Image(np.zeros(volume_shape), first_bold_nii.affine)

    print("Resampling ROI masks to match functional data...")
    resampled_roi_masks = {}
    for roi_name, roi_nii in original_roi_niftis.items():
        print(f"  - Resampling {roi_name}...")
        resampled_roi = resample_to_img(roi_nii, target_img_for_resampling, interpolation='nearest')
        resampled_roi_masks[roi_name] = resampled_roi.get_fdata().astype(bool)

    # <<< CHANGE START: Combine masks AFTER resampling
    if "ALL" in requested_rois:
        print("  - Creating 'ALL' mask from resampled components...")
        rois_to_combine = ['RSC', 'PPA', 'EVC', 'TOS']
        
        # Initialize an empty boolean mask with the correct shape
        combined_mask = np.zeros(volume_shape, dtype=bool)
        
        for component in rois_to_combine:
            if component in resampled_roi_masks:
                combined_mask = np.logical_or(combined_mask, resampled_roi_masks[component])
            else:
                print(f"    WARNING: Component '{component}' not found for creating 'ALL' mask.")
        
        resampled_roi_masks["ALL"] = combined_mask
    # <<< CHANGE END
    
    print("All masks prepared.")

    # --- 4. Mask data first, then aggregate trial data across all runs ---
    # Use defaultdict to easily append to lists
    session_data_per_roi = defaultdict(list)
    session_trial_events = []

    print(f"Found {len(run_bold_files)} runs. Applying masks and aggregating trial data...")
    for bold_file_path in run_bold_files:
        run_id = bold_file_path.split('run-')[1].split('_')[0]
        event_file_path = os.path.join(func_path, f'sub-{sub}_ses-{ses}_task-roomtour_run-{run_id}_events_aug.tsv')
        
        if event_file_path not in run_event_files:
            print(f"WARNING: Mismatch! BOLD file for run {run_id} found, but no event file. Skipping run.", file=sys.stderr)
            continue

        print(f"  - Processing run-{run_id}")
        bold_data = nib.load(bold_file_path).get_fdata()
        events_df = pd.read_csv(event_file_path, sep='\t')

        for index, trial in events_df.iterrows():
            onset_sec = trial['onset']
            start_tr = int(round(onset_sec / tr_seconds))
            end_tr = start_tr + num_trs_to_average

            if end_tr > bold_data.shape[3]:
                print(f"    WARNING: Trial at onset {onset_sec}s in run {run_id} exceeds BOLD data length. Skipping trial.", file=sys.stderr)
                continue
            
            # Extract the 4D volume for the current trial
            trial_trs_data = bold_data[:, :, :, start_tr:end_tr]
            
            # Now, apply each mask to this trial's data
            for roi_name, roi_mask in resampled_roi_masks.items():
                if roi_name == "ALL":
                    # Apply mask, resulting in a (n_voxels, n_trs) array
                    masked_trial_data = trial_trs_data[roi_mask]
                    # print(roi_name, "masked data shape:", masked_trial_data.shape)
                    
                    # flatten across the time dimension to get a feature vector for this trial
                    # trial_feature_vector = masked_trial_data.mean(axis=1)
                    trial_feature_vector = masked_trial_data
                    # print(roi_name, "feature vector shape:", trial_feature_vector.shape)
                    # sys.exit()
                    
                    # Append the feature vector for this trial to the correct ROI list
                    session_data_per_roi[roi_name].append(trial_feature_vector)
            
            # Append the event info. This happens once per trial, parallel to the data.
            session_trial_events.append(trial)

    if not session_trial_events:
        print(f"ERROR: No valid trials could be processed for sub-{sub}, ses-{ses}. Skipping session.", file=sys.stderr)
        return

    session_events = pd.DataFrame(session_trial_events).reset_index(drop=True)
    print(f"Data from all runs aggregated. Total trials: {len(session_events)}")

    # --- 5. Average trials by type and save features for each ROI ---
    unique_trial_types = session_events['trial_type'].unique()
    print(f"Found {len(unique_trial_types)} unique trial types. Averaging and saving features...")

    # Convert lists of trial data into numpy arrays for efficient indexing
    for roi_name in session_data_per_roi:
        # session_data_per_roi[roi_name] = np.vstack(session_data_per_roi[roi_name])
        session_data_per_roi[roi_name] = np.array(session_data_per_roi[roi_name])
        print("session_data_per_roi[roi_name].shape", session_data_per_roi[roi_name].shape)
        # sys.exit()

    for trial_type in unique_trial_types:
        print(f"  - Processing trial_type: {trial_type}")
        
        trial_indices = session_events.index[session_events['trial_type'] == trial_type].tolist()
        print("trial_indices", trial_indices)

        for roi_name, all_trials_data in session_data_per_roi.items():
            # Select the data for all occurrences of this trial type
            data_for_trial_type = all_trials_data[trial_indices, :]
            
            # Average across the trials to get the final mean feature vector
            mean_features = data_for_trial_type.mean(axis=0)

            # --- 6. Save the final feature vector ---
            output_dir = os.path.join(output_base_path, f'sub-{sub}', f'ses-{ses}')
            os.makedirs(output_dir, exist_ok=True)
            
            safe_trial_type = "".join(c for c in trial_type if c.isalnum() or c == '_').rstrip()
            output_filename = os.path.join(output_dir, f'sub-{sub}_ses-{ses}_{roi_name}_{safe_trial_type}.npy')
            
            np.save(output_filename, mean_features)
            print(f"    - Saved features for ROI '{roi_name}' ({mean_features.shape} voxels)")
            # sys.exit()

    print(f"Finished processing sub-{sub}, ses-{ses}.")

def load_output_files(output_base_path, sub, ses, roi, trial_type):
    output_dir = os.path.join(output_base_path, f'sub-{sub}', f'ses-{ses}')
    output_filename = os.path.join(output_dir, f'sub-{sub}_ses-{ses}_{roi}_{trial_type}.npy')
    data = np.load(output_filename)
    print(f"Loaded {output_filename} with {data.shape} features")

def main():
    """
    Main function to configure paths, define subjects/sessions,
    and run the processing pipeline.
    """
    # --- USER CONFIGURATION: PLEASE UPDATE THESE PATHS ---
    # NOTE: Use absolute paths for reliability.
    
    # Base path of your BIDS-like dataset
    # e.g., '/home/user/my_project/IDWB/ds_formal'
    DATA_BASE_PATH = '/mnt/d/IDWB/ds_formal/'
    
    # Path to the directory containing your ROI mask files
    # e.g., '/home/user/my_project/IDWB/scene_parcels'
    ROI_BASE_PATH = '/mnt/c/IDWB/scene_parcels'
    
    # Path where the output .npy files will be saved
    # e.g., '/home/user/my_project/IDWB/video-annotation/data/fmri-clip'
    OUTPUT_BASE_PATH = '/mnt/d/IDWB/video-annotation/data/fmri-clip-gd'

    # --- fMRI scan parameters ---
    TR_SECONDS = 1.0  # IMPORTANT: Set this to your scan's Repetition Time in seconds
    NUM_TRS_TO_AVERAGE = 5 # Number of TRs to average per trial, starting at onset
    
    # --- List of subjects and sessions to process ---
    # You can define them manually, e.g., SUBJECTS = ['01', '02']
    # Or discover them automatically from your data directory
    # try:
    #     SUBJECTS = sorted([os.path.basename(p) for p in glob.glob(os.path.join(DATA_BASE_PATH, 'sub-*'))])
    #     SUBJECTS = [s.replace('sub-', '') for s in SUBJECTS]
    #     if not SUBJECTS:
    #         raise FileNotFoundError
    # except FileNotFoundError:
    #     print(f"ERROR: Could not automatically find subjects in {DATA_BASE_PATH}. Please check the path.", file=sys.stderr)
    #     sys.exit(1)
    SUBJECTS = ['02']
        
    SESSIONS = ['01', '02'] # Example: ['01', '02'] or just ['01'] if there's only one

    # --- List of ROIs to process (SNR will be added automatically per subject) ---
    ROIS_TO_LOAD = ["ALL"]
    
    # --- PRE-PROCESSING ---
    # Load original ROI masks once to avoid reloading them for every subject
    original_roi_niftis = load_roi_masks(ROI_BASE_PATH, ROIS_TO_LOAD)
    if not original_roi_niftis:
        print("ERROR: Failed to load ROI masks. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- MAIN PROCESSING LOOP ---
    print("\nStarting fMRI feature extraction pipeline...")
    for sub in SUBJECTS:
        for ses in SESSIONS:
            try:
                # <<< CHANGE: Pass ROIS_TO_LOAD as the new argument
                process_subject_session(
                    sub, ses, DATA_BASE_PATH, original_roi_niftis, 
                    ROIS_TO_LOAD, # Pass the original user request
                    OUTPUT_BASE_PATH, TR_SECONDS, NUM_TRS_TO_AVERAGE
                )
            except Exception as e:
                print(f"FATAL ERROR processing sub-{sub}, ses-{ses}: {e}", file=sys.stderr)

    # load the output files
    load_output_files(OUTPUT_BASE_PATH, sub, '01', "ALL", "SPACE_06_MODN_030")
    load_output_files(OUTPUT_BASE_PATH, sub, '02', "ALL", "SPACE_11_MUJI_022")
    # load_output_files(OUTPUT_BASE_PATH, sub, ses, "PPA", "SPACE_11_MUJI_022")
    # load_output_files(OUTPUT_BASE_PATH, sub, ses, "TOS", "SPACE_11_MUJI_022")
    # load_output_files(OUTPUT_BASE_PATH, sub, ses, "RSC", "SPACE_11_MUJI_022")
    # load_output_files(OUTPUT_BASE_PATH, sub, ses, "SNR", "SPACE_11_MUJI_022")
    
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
