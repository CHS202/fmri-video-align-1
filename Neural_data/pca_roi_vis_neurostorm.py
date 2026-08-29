import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configure your ROI names and base path
BASE_PATH = ""
ROIS = ["EVC", "PPA", "RSC", "OPA"]  # Update with your actual ROI names
# FILENAME = "neurostorm_embeddings_pretrain_mae0.5_1.npy"

# Load all ROI data
print("Loading data...")
roi_data = {}

for roi in ROIS:
    if roi == "OPA":
        file_path = os.path.join(
            BASE_PATH, 
            "emotion_encoding_results/sub-01",
            f"neurostorm_pretrained_win4_TOS_1.npy"
        )
    else:
        file_path = os.path.join(
            BASE_PATH, 
            "emotion_encoding_results/sub-01",
            f"neurostorm_pretrained_win4_{roi}_1.npy"
        )
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        roi_data[roi] = data
        print(f"✓ Loaded {roi}: shape {data.shape}")
    else:
        print(f"✗ File not found: {file_path}")

if not roi_data:
    print("No files loaded. Please check your paths and ROI names.")
    exit(1)

# Concatenate all ROI data (744*4 = 2976 total samples)
print("\nConcatenating all ROI data...")
all_data = np.concatenate([roi_data[roi] for roi in ROIS], axis=0)
roi_labels = np.concatenate([np.full(roi_data[roi].shape[0], roi) for roi in ROIS])

print(f"Combined shape: {all_data.shape}")
print(f"Total samples: {len(roi_labels)}")

# Perform PCA on combined data
print("\nPerforming PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_data)

variance_explained = pca.explained_variance_ratio_
print(f"PC1 variance explained: {variance_explained[0]:.2%}")
print(f"PC2 variance explained: {variance_explained[1]:.2%}")

# Plot all samples from 4 ROIs on single PCA plot
fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#DD8452', '#55A868', '#C44E52', '#8172B2']  # Nice colors for 4 ROIs

for idx, roi in enumerate(ROIS):
    mask = roi_labels == roi
    ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
              label=f'{roi.upper()} (n={np.sum(mask)})', 
              alpha=0.6, 
              s=30, 
              color=colors[idx],
              edgecolors='none')

ax.set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=20)
ax.set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=20)
ax.set_title('PCA Distribution of Encoded fMRI', fontsize=24)
ax.legend(fontsize=20, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('pca_roi_comparison_neurostorm.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: pca_roi_comparison_neurostorm.png")
plt.show()

print("\n✅ Done!")