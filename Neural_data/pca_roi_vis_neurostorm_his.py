import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configure your ROI names and base path
BASE_PATH = ""
ROIS = ["EVC", "OPA", "PPA", "RSC"]  # Update with your actual ROI names
N_COMPONENTS = 12  # <-- increase this to see more PCs (was 2)

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

    d = roi_data[roi]
    per_feature_std = d.std(axis=0)  # variability across the 720 samples, per feature
    print(f"{roi}: per-feature std across samples — "
          f"mean={per_feature_std.mean():.5f}, max={per_feature_std.max():.5f}")

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
print(f"\nPerforming PCA with {N_COMPONENTS} components...")
pca = PCA(n_components=N_COMPONENTS)
pca_result = pca.fit_transform(all_data)

variance_explained = pca.explained_variance_ratio_
for i in range(N_COMPONENTS):
    print(f"PC{i+1} variance explained: {variance_explained[i]:.2%}")

# --- Debug: check the actual spread of each ROI along each PC ---
for idx, roi in enumerate(ROIS):
    mask = roi_labels == roi
    ranges = [f"PC{i+1} [{pca_result[mask, i].min():.2f}, {pca_result[mask, i].max():.2f}]"
              for i in range(N_COMPONENTS)]
    print(f"{roi}: " + ", ".join(ranges))

# --- Plot with shared bins so no ROI gets visually swallowed ---
# Dynamic grid: pick number of columns, compute rows from N_COMPONENTS
n_cols = min(4, N_COMPONENTS)
n_rows = int(np.ceil(N_COMPONENTS / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
axes = np.atleast_1d(axes).flatten()  # handles n_rows*n_cols == 1 case too

colors = ['#DD8452', '#55A868', '#C44E52', '#8172B2']
n_bins = 40

for pc_idx in range(N_COMPONENTS):
    ax = axes[pc_idx]
    # shared bin edges across the full range of this PC (all ROIs combined)
    pc_min, pc_max = pca_result[:, pc_idx].min(), pca_result[:, pc_idx].max()
    bin_edges = np.linspace(pc_min, pc_max, n_bins + 1)

    for idx, roi in enumerate(ROIS):
        mask = roi_labels == roi
        ax.hist(pca_result[mask, pc_idx],
                bins=bin_edges,
                label=f'{roi.upper()} (n={np.sum(mask)})',
                alpha=0.5,
                color=colors[idx],
                edgecolor=colors[idx],
                linewidth=1.0)

    ax.set_xlabel(f'PC{pc_idx + 1}', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Distribution along PC{pc_idx + 1}', fontsize=16)
    # ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

# Hide any unused subplot axes (e.g. if N_COMPONENTS doesn't fill the grid)
for extra_idx in range(N_COMPONENTS, len(axes)):
    axes[extra_idx].set_visible(False)

# Build one shared legend from the first subplot's handles (same across all subplots)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=len(ROIS), fontsize=14, framealpha=0.95)

fig.suptitle('PCA Distribution of Encoded fMRI', fontsize=24, y=1.02)
plt.tight_layout()
plt.savefig('pca_roi_comparison_neurostorm_hist.png', dpi=300, bbox_inches='tight')
plt.show()