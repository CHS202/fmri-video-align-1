import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import h5py

sub = '01'
split = '1'
roi = 'EVC'

video_id_to_name = {}
df = pd.read_csv("/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv")
with open("/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) # Skip header
    for row in reader:
        video_id_to_name[int(row[0])] = row[1]

video_order = np.array(h5py.File(f'video_order_rt_{split}.mat')['video_order'])

all_embeddings = np.load(f"emotion_encoding_results/sub-{sub}/voxel_select_new_{roi}_{split}.npy")
# check if nan exists
if np.isnan(all_embeddings).any():
    print("Nan exists in all_embeddings")

design_to_label = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}
label_to_design = {v: k for k, v in design_to_label.items()}
design_names = [label_to_design[i] for i in range(4)]  # ['MODN', 'MUJI', 'SCAN', 'WABI']

all_targets = np.array([design_to_label[video_id_to_name[i].split('/')[0]] for i in video_order])

# --- Average embeddings per design label ---
averaged_embeddings = np.stack([
    all_embeddings[all_targets == i].mean(axis=0)
    for i in range(4)
])  # Shape: (4, embedding_dim)
print(averaged_embeddings[:2, :2])

# --- Compute 4x4 correlation matrix ---
correlation_matrix = np.corrcoef(averaged_embeddings)  # Shape: (4, 4)
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# --- Plotting ---
plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, vmin=0.995, vmax=1, cmap='coolwarm')
plt.colorbar(label='Correlation Coefficient')

plt.xticks(np.arange(4), design_names, rotation=45, fontsize=10)
plt.yticks(np.arange(4), design_names, fontsize=10)

# Annotate each cell with the correlation value
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{correlation_matrix[i, j]:.5f}",
                    ha='center', va='center', fontsize=10, color='black')

plt.title('Design Label Correlation Matrix')
plt.tight_layout()
plt.savefig(f"emotion_encoding_results/sub-{sub}/voxel_select_new_corr_matrix_{roi}_split{split}.png")