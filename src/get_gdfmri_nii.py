import os
import pandas as pd
import nibabel as nib
import numpy as np

def split_nifi_by_events(
    nii_path, 
    tsv_path, 
    output_dir, 
    subject, 
    ses_idx, 
    run_idx,
    # hrf_delay_sec=4.0
):
    """
    Splits a 4D NIfTI file into smaller NIfTI files based on a TSV event file.
    """
    
    # 1. Load the Data
    print(f"Loading NIfTI: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata() # Load data into memory
    header = img.header
    affine = img.affine
    
    print(f"Loading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')

    # 2. Get Repetition Time (TR) to convert Seconds -> Frames
    # usually stored in header.get_zooms()[3]
    try:
        tr = header.get_zooms()[3]
    except IndexError:
        tr = 0
        
    # Safety check: If TR is 0 or missing in header, you must provide it manually
    if tr == 0:
        raise ValueError("TR (Repetition Time) not found in NIfTI header. Please specify TR manually in the code.")
    
    print(f"Detected TR: {tr} seconds")
    # print how many trs
    print(f"Number of TRs: {data.shape[3]}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. Iterate through the TSV
    count = 0
    for index, row in df.iterrows():
        onset_sec = row['onset']
        duration_sec = row['duration']
        trial_type = row['trial_type']
        
        # 4. Convert Time (seconds) to Indices (frames)
        # We use round() to handle floating point precision issues
        start_frame = int(np.round(onset_sec / tr))
        end_frame = int(np.round((onset_sec + duration_sec) / tr))
        
        # Validate bounds
        if start_frame < 0 or end_frame > data.shape[3]:
            print(f"Warning: Trial {trial_type} at {onset_sec}s is out of bounds. Skipping.")
            continue

        # 5. Slice the data
        # Data shape is (x, y, z, time)
        sliced_data = data[..., start_frame:end_frame]
        
        # 6. Create new NIfTI image
        # We allow the header to update automatically for the new 4th dimension size
        new_img = nib.Nifti1Image(sliced_data, affine, header)
        
        # Update the header to reflect the new number of timepoints
        new_img.header['dim'][4] = sliced_data.shape[3]

        # 7. Construct Filename
        # Format: sub-{subject}_ses-{ses_idx}_task-roomtour_run-{run_idx}_{trial_type}.nii.gz
        # We sanitize trial_type just in case it has spaces or weird characters
        safe_trial_name = str(trial_type).strip().replace(" ", "_")
        
        filename = f"sub-{subject}_ses-{ses_idx}_task-roomtour_run-{run_idx}_{safe_trial_name}.nii.gz"
        out_path = os.path.join(output_dir, filename)
        
        # 8. Save
        nib.save(new_img, out_path)
        count += 1

    print(f"Successfully created {count} files in '{output_dir}'")

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Define your paths here
    # You can replace these with your actual file paths
    subject = ['01', '02', '03', '06', '08', '09', '12']
    ses_idxs = ["01", "02"]
    run_idxs = ["01", "02"]
    
    for sub in subject:
        for ses_idx in ses_idxs:
            for run_idx in run_idxs:
                # Create dummy files for demonstration (You don't need this block if you have real files)
                # ----------------------------------------------------------------
                # (This is just to make the script runnable for testing purposes)
                nii_name = f"/mnt/d/IDWB/ds_formal/derivatives/glm_denoise_results/sub-{sub}/ses-{ses_idx}/sub-{sub}_ses-{ses_idx}_task-roomtour_run-{run_idx}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
                tsv_name = f"/mnt/d/IDWB/ds_formal/sub-{sub}/ses-{ses_idx}/func/sub-{sub}_ses-{ses_idx}_task-roomtour_run-{run_idx}_events_aug.tsv"
                # ----------------------------------------------------------------

                # RUN THE FUNCTION
                split_nifi_by_events(
                    nii_path=nii_name,           # Path to your .nii or .nii.gz file
                    tsv_path=tsv_name,           # Path to your .tsv file
                    output_dir=f"/mnt/d/IDWB/ds_formal/derivatives/gdfmri_clips/sub-{sub}/ses-{ses_idx}/",      # Where to save the new files
                    subject=sub,                  # sub-01
                    ses_idx=ses_idx,                      # ses-01
                    run_idx=run_idx,                      # run-01
                    # hrf_delay_sec=4.0
                )