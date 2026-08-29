import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configure your ROI names and base path
ROIS = ["EVC", "OPA", "PPA", "RSC"]  # Your ROI names
FILE_PATHS = {
    "EVC": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_EVC_1.npy",
    "OPA": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_TOS_1.npy",
    "PPA": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_PPA_1.npy",
    "RSC": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_RSC_1.npy",
}

N_COMPONENTS = 12  # <-- change this to control how many PCs are computed/plotted

# Load all ROI data
print("Loading data...")
roi_data = {}

for roi in ROIS:
    file_path = FILE_PATHS[roi]
    
    if os.path.exists(file_path):
        data = np.load(file_path)
        roi_data[roi] = data
        print(f"✓ Loaded {roi}: shape {data.shape}")
    else:
        print(f"✗ File not found: {file_path}")

if not roi_data:
    print("No files loaded. Please check your file paths.")
    exit(1)

print("\n" + "="*50)
print("Method: Each ROI gets individual PCA reduction")
print("="*50)

# Reduce each ROI separately using PCA
# Then concatenate and visualize together
print(f"\nPerforming PCA on each ROI separately ({N_COMPONENTS} components)...")
pca_reduced_data = {}
pca_models = {}

for roi in ROIS:
    data = roi_data[roi]
    pca = PCA(n_components=N_COMPONENTS)
    reduced = pca.fit_transform(data)
    pca_reduced_data[roi] = reduced
    pca_models[roi] = pca
    
    variance_explained = pca.explained_variance_ratio_
    var_str = ", ".join([f"PC{i+1}={v:.1%}" for i, v in enumerate(variance_explained)])
    print(f"{roi}: {var_str}")

# Plot PC distributions as histograms, one subplot per component
n_cols = min(4, N_COMPONENTS)
n_rows = int(np.ceil(N_COMPONENTS / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
axes = np.atleast_1d(axes).flatten()

colors = ['#DD8452', '#55A868', '#C44E52', '#8172B2']
n_bins = 40

for pc_idx in range(N_COMPONENTS):
    ax = axes[pc_idx]
    for idx, roi in enumerate(ROIS):
        pca_result = pca_reduced_data[roi]
        ax.hist(pca_result[:, pc_idx],
                bins=n_bins,
                label=f'{roi} (n={len(pca_result)})',
                alpha=0.5,
                color=colors[idx],
                edgecolor=colors[idx],
                linewidth=1.0)

    ax.set_xlabel(f'PC{pc_idx + 1}', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Distribution along PC{pc_idx + 1}', fontsize=16)
    # ax.legend(fontsize=9, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

# Hide unused grid cells if any
for extra_idx in range(N_COMPONENTS, len(axes)):
    axes[extra_idx].set_visible(False)

# Build one shared legend from the first subplot's handles (same across all subplots)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=len(ROIS), fontsize=14, framealpha=0.95)

fig.suptitle('PCA Distribution of Denoised fMRI', fontsize=24, y=1.02)
plt.tight_layout()
plt.savefig('pca_roi_comparison_hist.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: pca_roi_comparison_hist.png")
plt.show()

print("\n✅ Done!")