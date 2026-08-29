import pandas as pd
import h5py
import numpy as np

# Load CSVs
df_all = pd.read_csv('../BrainGuided/video_id_rt.csv')
df_annot = pd.read_csv('../BrainGuided/video_id_rt_annot.csv')

# Create mappings
annot_id_to_name = dict(zip(df_annot['Video ID'], 
                            df_annot['Video Name and Directory']))
name_to_all_id = dict(zip(df_all['Video Name and Directory'], 
                          df_all['Video ID']))
annot_to_all_id = {aid: name_to_all_id[annot_id_to_name[aid]] 
                   for aid in df_annot['Video ID']}

for sub in ["01", "02", "03", "06", "08", "09", "12"]:
    for split in [1, 2, 3, 4]:
        for roi in ["PFC"]:
            # Load NPY and video orders
            # npy_data = np.load(f'emotion_encoding_results/sub-{sub}/voxel_select_new_valid_{roi}_{split}.npy')
            # npy_data = np.load(f'emotion_encoding_results/sub-{sub}/neurostorm_pretrained_win4_{roi}_{split}.npy')
            npy_data = np.load(f'emotion_encoding_results/sub-{sub}/neurostorm_pretrained_win4_valid_{roi}_{split}.npy')

            with h5py.File(f'video_order_rt_valid_{split}.mat', 'r') as f:
                order_all = f['video_order'][:].astype(int)

            with h5py.File(f'video_order_rt_annot_valid_{split}.mat', 'r') as f:
                annot_ids = f['video_order'][:].astype(int)  # Annotation CSV IDs!

            # Create position mapping
            all_id_to_position = {vid: pos for pos, vid in enumerate(order_all)}

            # Convert: annotation_id → all_videos_id → position
            annot_positions = [all_id_to_position[annot_to_all_id[aid]] 
                            for aid in annot_ids]

            # Extract
            annot_data = npy_data[annot_positions]
            # Save
            np.save(f'emotion_encoding_results/sub-{sub}/neurostorm_pretrained_win4_annot_valid_{roi}_{split}.npy', annot_data)
            print(f'Finished sub-{sub}, split-{split}, shape: {annot_data.shape}')