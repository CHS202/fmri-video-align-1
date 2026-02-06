import os
import csv
import numpy as np
import h5py
import sys

def find_voxel_file_path(base_name, roi, subject_dir, video_name):
    """Searches for a voxel file in session 1 or 2."""
    # remove "clip" in video_name
    video_name = video_name.replace("_clip", "")
    for session in ["ses-01", "ses-02"]:
        file_path = os.path.join(subject_dir, session, f"{base_name}_{session}_{roi}_{video_name}.npy")
        if os.path.exists(file_path):
            return file_path
    return None


def generate_voxel_selection(config):
    """
    Main function to generate the voxel selection npy file.
    """
    # --- 1. Load video order and create ID-to-Name mapping ---
    print(f"Loading video order from {config['mat_order_file']} and ID-to-name mapping...")
    try:
        video_order = np.array(h5py.File(config['mat_order_file'])['video_order'])
    except FileNotFoundError:
        print(f"Error: Input file not found: '{config['mat_order_file']}'")
        return
        
    try:
        video_id_to_name = {}
        with open(config['csv_file'], 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                video_id_to_name[int(row[0])] = row[1]
    except FileNotFoundError:
        print(f"Error: Input file not found: '{config['csv_file']}'")
        return

    # --- 2. Process each video in the specified order ---
    print("Processing voxel data for each video...")
    all_videos_concatenated_voxels = []
    total_voxel_size = -1 # To be determined from the first video

    for roi in config['rois']:
        voxels_for_video = []
        for video_id in video_order:
            video_rel_path = video_id_to_name.get(video_id)
            if not video_rel_path:
                print(f"Warning: Video ID {video_id} from .mat file not found in CSV. Skipping.")
                continue
            
            # Extract base name, e.g., "SPACE_05_MODN_clip_000"
            video_base_name = os.path.basename(video_rel_path).replace('.mp4', '')
            
            voxel_path = find_voxel_file_path(
                config['subject_base_name'], roi, config['fmri_subject_dir'], video_base_name
            )
            
            if voxel_path:
                roi_data = np.load(voxel_path)
                # print(f"Loaded {voxel_path} with {roi_data.shape} features")
                # sys.exit()
                # if roi == 'ALL':
                roi_data = roi_data.T
                voxels_for_video.append(roi_data)
            else:
                print(f"Warning: Voxel file not found for video '{video_base_name}', ROI '{roi}'. Skipping video.")
                voxels_for_video = [] # Clear list to skip this video
                break
        
        # If all ROIs were found, concatenate and add them
        # print("shape of voxels_for_video", voxels_for_video.shape)
        # sys.exit()
        # if voxels_for_video:
        #     # if roi == 'ALL':
        #     concatenated_voxels = np.concatenate(voxels_for_video)
            
        #     # Check for consistent voxel counts
        #     if total_voxel_size == -1:
        #         total_voxel_size = len(concatenated_voxels)
        #     elif len(concatenated_voxels) != total_voxel_size:
        #         print(f"Warning: Inconsistent voxel size for video '{video_base_name}'. Skipping.")
        #         continue

        #     all_videos_concatenated_voxels.append(concatenated_voxels)

        # # --- 3. Save the final array ---
        # if not all_videos_concatenated_voxels:
        #     print("Error: No voxel data was successfully processed. Output file will not be generated.")
        #     return

        # final_array = np.array(all_videos_concatenated_voxels)
        final_array = np.array(voxels_for_video)
        print(f"\nFinal array shape: {final_array.shape}")

        os.makedirs(config['output_dir'], exist_ok=True)
        output_path = os.path.join(config['output_dir'], f"voxel_select_remain_time_{roi}_{SPLIT}.npy")
        np.save(output_path, final_array)
        print(f"Successfully saved voxel selection to '{output_path}'")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    # NOTE: For the dummy setup, we create 'fmri-clip' locally.
    # For your real data, change this path to the parent directory, e.g., 'D:\IDWB\video-annotation\data'
    FMRI_DATA_ROOT = "/mnt/d/IDWB/video-annotation/data/fmri-clip-gd" 
    SUBJECT_ID = "02"
    SPLIT = 1
    for SPLIT in [1, 2, 3, 4]:
        config = {
            "csv_file": "../BrainGuided/video_id_rt.csv",
            "mat_order_file": f"video_order_rt_{SPLIT}.mat",
            "fmri_subject_dir": os.path.join(FMRI_DATA_ROOT, f"sub-{SUBJECT_ID}"),
            "subject_base_name": f"sub-{SUBJECT_ID}",
            "rois": ['ALL','EVC', 'PPA', 'RSC', 'TOS'],
            # "rois": ['ALL'],
            "output_dir": f"emotion_encoding_results/sub-{SUBJECT_ID}",
            # "output_file_name": f"voxel_select_remain_time_{SPLIT}.npy"
        }

        # 2. Run the main function to process the data and generate the output.
        generate_voxel_selection(config)

        # --- 2. NEW Code: Average time in 'voxel_select_remain_time_*.npy' ---
        for roi in config['rois']:
            print(f"--- Averaging time dimension for split {SPLIT} ---")

            # Define the input and output file paths using your config
            input_dir = config['output_dir']
            input_filename = f"voxel_select_remain_time_{roi}_{SPLIT}.npy"
            output_filename = f"voxel_select_new_{roi}_{SPLIT}.npy"
            
            input_path = os.path.join(input_dir, input_filename)
            output_path = os.path.join(input_dir, output_filename)

            try:
                # Load the data
                print(f"Loading data from: {input_path}")
                data = np.load(input_path)
                print(f"Original data shape: {data.shape}") # Should be (744, 5, 7545)

                # Average along the time dimension (axis=1)
                # Shape (744, 5, 7545) -> (744, 7545)
                averaged_data = np.mean(data, axis=1)
                print(f"Averaged data shape: {averaged_data.shape}")

                # Save the new array
                np.save(output_path, averaged_data)
                print(f"Successfully saved averaged data to: {output_path}")

            except FileNotFoundError:
                print(f"Error: Input file not found at '{input_path}'.")
                print(f"Please make sure 'voxel_select_remain_time_{roi}_{SPLIT}.npy' exists.")
            except Exception as e:
                print(f"An error occurred during averaging: {e}")
