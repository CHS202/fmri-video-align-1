import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configure your ROI names and base path
ROIS = ["EVC", "PPA", "RSC", "OPA"]  # Your ROI names
FILE_PATHS = {
    "EVC": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_EVC_1.npy",
    "PPA": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_PPA_1.npy",
    "RSC": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_RSC_1.npy",
    "OPA": "/mnt/d/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-01/voxel_select_new_TOS_1.npy"
}

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

# Reduce each ROI to 2D using PCA separately
# Then concatenate and visualize together
print("\nPerforming PCA on each ROI separately...")
pca_reduced_data = {}
pca_models = {}

for roi in ROIS:
    data = roi_data[roi]
    # Reduce to 2 components for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    pca_reduced_data[roi] = reduced
    pca_models[roi] = pca
    
    variance_explained = pca.explained_variance_ratio_
    print(f"{roi}: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}")

# Plot all ROIs on single figure
fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#DD8452', '#55A868', '#C44E52', '#8172B2']
alpha_val = 0.6
s_val = 40

for idx, roi in enumerate(ROIS):
    pca_result = pca_reduced_data[roi]
    ax.scatter(pca_result[:, 0], pca_result[:, 1], 
              label=f'{roi} (n={len(pca_result)})', 
              alpha=alpha_val, 
              s=s_val, 
              color=colors[idx],
              edgecolors='none')

ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
ax.set_title('PCA Distribution of Denoised fMRI', 
             fontsize=24)
ax.legend(fontsize=20, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('pca_roi_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: pca_roi_comparison.png")
plt.show()

# # Alternative: Reduce each ROI to same number of components, then combine
# print("\n" + "="*50)
# print("Alternative Method: Reduce to common dimension, then combine")
# print("="*50)

# # Reduce each ROI to 10 components first
# n_components_first = 10
# print(f"\nReducing each ROI to {n_components_first} components first...")

# all_reduced = []
# roi_labels = []

# for roi in ROIS:
#     data = roi_data[roi]
#     pca = PCA(n_components=min(n_components_first, data.shape[1], data.shape[0]))
#     reduced = pca.fit_transform(data)
#     all_reduced.append(reduced)
#     roi_labels.extend([roi] * len(reduced))
#     print(f"{roi}: {data.shape} -> {reduced.shape}")

# # Concatenate all reduced data
# combined_reduced = np.vstack(all_reduced)
# print(f"\nCombined shape: {combined_reduced.shape}")

# # Final PCA for visualization
# print("Performing final PCA on combined data...")
# final_pca = PCA(n_components=2)
# final_pca_result = final_pca.fit_transform(combined_reduced)

# variance_final = final_pca.explained_variance_ratio_
# print(f"Final PC1: {variance_final[0]:.1%}, PC2: {variance_final[1]:.1%}")

# # Plot combined
# fig, ax = plt.subplots(figsize=(14, 10))

# for idx, roi in enumerate(ROIS):
#     mask = np.array(roi_labels) == roi
#     ax.scatter(final_pca_result[mask, 0], final_pca_result[mask, 1], 
#               label=f'{roi} (n={np.sum(mask)})', 
#               alpha=alpha_val, 
#               s=s_val, 
#               color=colors[idx],
#               edgecolors='none')

# ax.set_xlabel(f'PC1 ({variance_final[0]:.1%})', fontsize=14, fontweight='bold')
# ax.set_ylabel(f'PC2 ({variance_final[1]:.1%})', fontsize=14, fontweight='bold')
# ax.set_title('PCA Distribution of 4 ROIs\n(Reduced to 10D, then combined and visualized)', 
#              fontsize=20, fontweight='bold')
# ax.legend(fontsize=14, loc='best', framealpha=0.95)
# ax.grid(True, alpha=0.3, linestyle='--')

# plt.tight_layout()
# plt.savefig('pca_roi_comparison_method2.png', dpi=300, bbox_inches='tight')
# print("\n✓ Saved: pca_roi_comparison_method2.png")
# plt.show()

print("\n✅ Done! Compare both methods to see which fits your analysis better.")