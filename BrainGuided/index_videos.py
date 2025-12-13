import os
import csv
import h5py
import numpy as np

def generate_video_csv(root_dir, subdirs, output_csv_path):
    """
    Scans subdirectories for .mp4 files and generates a CSV file mapping
    a unique ID to each video's relative path.

    Args:
        root_dir (str): The main directory containing the video subdirectories.
        subdirs (list): A list of subdirectory names to scan for videos.
        output_csv_path (str): The file path for the generated CSV.
    """
    video_data = []
    video_id_counter = 1

    print(f"Scanning for videos in '{root_dir}'...")

    # Iterate through the specified subdirectories
    for subdir in subdirs:
        current_dir_path = os.path.join(root_dir, subdir)

        # Check if the subdirectory actually exists
        if not os.path.isdir(current_dir_path):
            print(f"Warning: Directory '{current_dir_path}' not found. Skipping.")
            continue

        # List all files in the current subdirectory and sort them for consistent order
        try:
            for filename in sorted(os.listdir(current_dir_path)):
                # Process only if the file is an .mp4 video
                if filename.lower().endswith(".mp4"):
                    # Create the relative path string as required
                    video_name_and_dir = f"{subdir}/{filename}"

                    # Append the new row to our data list
                    video_data.append([video_id_counter, video_name_and_dir])

                    # Increment the ID for the next video
                    video_id_counter += 1
        except OSError as e:
            print(f"Error accessing directory '{current_dir_path}': {e}")


    # Write the collected video data to the CSV file
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row
            writer.writerow(["Video ID", "Video Name and Directory"])

            # Write all the video data rows
            writer.writerows(video_data)

        print(f"\nSuccessfully generated '{output_csv_path}' with {len(video_data)} video entries.")
    except IOError as e:
        print(f"Error: Could not write to file '{output_csv_path}': {e}")

# def generate_train_test_split(csv_path, train_output_path, test_output_path, mat_output_path):
#     """
#     Reads the video ID CSV and splits the IDs into training and testing sets.
#     The split is based on video names, and both sets are shuffled before saving
#     as .npy files.

#     Args:
#         csv_path (str): Path to the input video_id_rt.csv file.
#         train_output_path (str): Path to save the training index .npy file.
#         test_output_path (str): Path to save the testing index .npy file.
#     """
#     train_ids = []
#     test_ids = []
#     test_video_keys = ["SPACE_05", "SPACE_09"]

#     print(f"\n--- Generating Train/Test Split from '{csv_path}' ---")

#     try:
#         with open(csv_path, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             next(reader)  # Skip the header row

#             for row in reader:
#                 video_id = int(row[0])
#                 video_name = row[1]

#                 # Check if any of the test keys are in the video name
#                 if any(key in video_name for key in test_video_keys):
#                     test_ids.append(video_id)
#                 else:
#                     train_ids.append(video_id)

#         # Convert to numpy arrays
#         train_ids = np.array(train_ids, dtype=np.int32)
#         test_ids = np.array(test_ids, dtype=np.int32)


#         # Shuffle the arrays
#         # np.random.shuffle(train_ids)
#         # np.random.shuffle(test_ids)

#         # Save the arrays to .npy files
#         np.save(train_output_path, train_ids)
#         np.save(test_output_path, test_ids)

#         # Save the training order to a .mat file for MATLAB/h5py compatibility
#         with h5py.File(mat_output_path, 'w') as hf:
#             hf.create_dataset('video_order', data=train_ids)

#         print(f"Training set size: {len(train_ids)}")
#         print(f"Testing set size: {len(test_ids)}")
#         print(f"Successfully created '{train_output_path}', '{test_output_path}', and '{mat_output_path}'.")

#     except FileNotFoundError:
#         print(f"Error: Could not find '{csv_path}'. Please generate it first.")
#     except Exception as e:
#         print(f"An error occurred during train/test split generation: {e}")

def generate_4_splits(csv_path, train_output_path, test_output_path):
    """
    Reads video IDs from a CSV and generates 4 distinct train/test splits.

    This version saves each split's training order into a separate .mat file
    (e.g., train_order_split_0.mat, train_order_split_1.mat, etc.).

    Args:
        csv_path (str): Path to the input video_id_rt.csv file.
        output_dir (str): Directory to save the output .npy and .mat files.
    """
    # --- IMPORTANT ---
    # You MUST customize this list based on your specific video names.
    all_test_keys = [
        ["SPACE_05", "SPACE_09", "SPACE_01"],  # Test keys for Split 1
        ["SPACE_06", "SPACE_10", "SPACE_02"],  # Test keys for Split 2
        ["SPACE_07", "SPACE_11", "SPACE_03"],  # Test keys for Split 3
        ["SPACE_08", "SPACE_12", "SPACE_04"]   # Test keys for Split 4
    ]

    print(f"\n--- Generating 4 Train/Test Splits from '{csv_path}' ---")

    try:

        # 1. Read all video data from CSV once
        all_video_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                all_video_data.append((int(row[0]), row[1]))

        all_train_splits = []
        all_test_splits = []

        # 2. Loop through the keys and generate data for each split
        for i, test_keys in enumerate(all_test_keys):
            print(f"--- Processing Split {i+1}/4 (Test keys: {test_keys}) ---")
            train_ids = []
            test_ids = []

            for video_id, video_name in all_video_data:
                if any(key in video_name for key in test_keys):
                    test_ids.append(video_id)
                else:
                    train_ids.append(video_id)

            train_ids_np = np.array(train_ids, dtype=np.int32)
            test_ids_np = np.array(test_ids, dtype=np.int32)

            all_train_splits.append(train_ids_np)
            all_test_splits.append(test_ids_np)
            
            print(f"Split {i+1}: Train size = {len(train_ids)}, Test size = {len(test_ids)}")

            # --- MODIFICATION START ---
            # 3. Save the .mat file for the CURRENT split inside the loop
            # mat_output_path = f'video_order_rt_{i+1}.mat'
            # with h5py.File(mat_output_path, 'w') as hf:
            #     # Since the split is identified by the filename,
            #     # we can use a consistent dataset name inside.
            #     hf.create_dataset('video_order', data=train_ids_np)
            # print(f"✅ Saved training order to '{mat_output_path}'")
            # --- MODIFICATION END ---


        # 4. Convert lists of arrays into a single NumPy array of objects
        final_train_splits = np.array(all_train_splits)
        final_test_splits = np.array(all_test_splits)

        # 5. Define .npy output paths and save the final .npy files
        # train_output_path = 'train_idx_rt.npy'
        # test_output_path = 'test_idx_rt.npy'
        
        np.save(train_output_path, final_train_splits)
        np.save(test_output_path, final_test_splits)

        print(f"\n✅ Saved all training splits to '{train_output_path}'")
        print(f"✅ Saved all testing splits to '{test_output_path}'")

    except FileNotFoundError:
        print(f"❌ Error: Could not find '{csv_path}'. Please check the file path.")
    except Exception as e:
        print(f"❌ An error occurred during split generation: {e}")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    ROOT_VIDEO_DIRECTORY = "RT--raw"
    SUBDIRECTORIES_TO_SCAN = ["MODN", "MUJI", "WABI", "SCAN"]
    OUTPUT_CSV_FILE = "video_id_rt_ses123.csv"
    TRAIN_IDX_FILE = "train_idx_rt_ses123.npy"
    TEST_IDX_FILE = "test_idx_rt_ses123.npy"
    # MAT_ORDER_FILE = "video_order_rt.mat"

    # 2. Run the main function to generate the CSV.
    generate_video_csv(ROOT_VIDEO_DIRECTORY, SUBDIRECTORIES_TO_SCAN, OUTPUT_CSV_FILE)

    # 3. Generate the train/test split .npy files from the CSV.
    # generate_train_test_split(OUTPUT_CSV_FILE, TRAIN_IDX_FILE, TEST_IDX_FILE, MAT_ORDER_FILE)
    generate_4_splits(OUTPUT_CSV_FILE, TRAIN_IDX_FILE, TEST_IDX_FILE)

    # 4. (Optional) Print the content of the generated files for verification.
    print(f"\n--- Contents of '{OUTPUT_CSV_FILE}' ---")
    try:
        with open(OUTPUT_CSV_FILE, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("Could not read the output file.")

    print(f"\n--- Contents of '{TRAIN_IDX_FILE}' and '{TEST_IDX_FILE}'---")
    try:
        train_data = np.load(TRAIN_IDX_FILE)
        test_data = np.load(TEST_IDX_FILE)
        print(f"Training IDs: {train_data}, {train_data.shape}")
        print(f"Testing IDs: {test_data}, {test_data.shape}")
    except FileNotFoundError:
        print("Could not read the .npy output files.")

    
    # for i in range(1,5):
    #     print(f"\n--- Contents of train_order_rt_{i}.mat (read using h5py) ---")
    #     mat_output_path = f'video_order_rt_{i}.mat'
    #     try:
    #         with h5py.File(mat_output_path, 'r') as f:
    #             video_order = np.array(f['video_order'])
    #             print(f"Video Order (.mat): {video_order}")
    #     except FileNotFoundError:
    #         print(f"Could not read the '{mat_output_path}' output file.")
