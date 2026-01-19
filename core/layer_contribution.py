import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def calculate_layer_contributions(all_labels, layers_output, opt):
    """
    1. PCA on each layer.
    2. Concatenate all PCA-reduced features.
    3. Fit ONE Ridge Regression.
    4. Calculate C_l = cov(phi_l * W_l, y) / (var(phi_l * W_l) * var(y)) per layer.
    """
    
    reduced_features_list = []
    layer_metadata = [] # To keep track of which columns belong to which layer
    current_col_idx = 0

    # --- PCA per layer and prepare for concatenation ---
    # print("Performing PCA and preparing features...")
    for key, feats in layers_output.items():
        # Move to CPU and flatten
        feats = feats.detach().cpu().numpy()
        if len(feats.shape) > 2:
            feats_flat = feats.reshape(feats.shape[0], -1)
        else:
            feats_flat = feats

        # PCA: keep 99% variance
        pca = PCA(n_components=0.99)
        feats_pca = pca.fit_transform(feats_flat)
        
        # Scale the PCA components so layers are on a comparable scale
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats_pca)
        
        n_components = feats_scaled.shape[1]
        reduced_features_list.append(feats_scaled)
        
        # Store indices: (layer_name, start_index, end_index)
        layer_metadata.append({
            'name': key,
            'start': current_col_idx,
            'end': current_col_idx + n_components
        })
        current_col_idx += n_components

    # Concatenate all features into one matrix (N_samples x Total_PCA_Components)
    X_combined = np.hstack(reduced_features_list)
    y = all_labels # Shape: (N_samples, N_voxels)
    # print(f"Combined features shape: {X_combined.shape}")
    # print(f"Labels shape: {y.shape}")
    
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # --- Run Ridge Regression ---
    # print(f"Fitting Ridge Regression on combined features (Shape: {X_combined.shape})...")
    # alphas can be adjusted based on your needs
    y = y.detach().cpu().numpy()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
    ridge.fit(X_combined, y)
    
    # ridge.coef_ has shape (N_voxels, Total_PCA_Components)
    all_weights = ridge.coef_

    # --- Calculate TOTAL prediction (all layers combined) ---
    # pred_all shape: (N_samples, N_voxels)
    pred_all = X_combined @ all_weights.T 
    # calculate prediction mse
    mse = np.mean((pred_all - y) ** 2, axis=0)
    mse = np.mean(mse, axis=0)
    # r-square
    r2 = 1 - mse / np.var(y, axis=0)
    r2 = np.mean(r2, axis=0)

    # --- Calculate Layer Contributions based on reference formula ---
    contributions = {}
    
    for meta in layer_metadata:
        name = meta['name']
        start, end = meta['start'], meta['end']
        
        # Extract specific layer features and weights
        phi_l = X_combined[:, start:end] 
        W_l = all_weights[:, start:end]  
        
        # Calculate layer-specific prediction (val_pred_lw)
        pred_l = phi_l @ W_l.T 
        
        layer_v_contribs = []
        
        for v in range(y.shape[1]):
            # Get data for this specific voxel
            y_v = y[:, v]
            p_full_v = pred_all[:, v]
            p_layer_v = pred_l[:, v]
            
            # 1. Calculate covariance of Layer Prediction with Y (Numerator)
            cov_layer_y = np.cov(p_layer_v, y_v, ddof=0)[0, 1]
            
            # 2. Calculate variance of Full Prediction and Y (Denominator)
            # This matches: np.sqrt(full_c[0,0] * full_c[1,1]) from your snippet
            var_full = np.var(p_full_v, ddof=0)
            var_y = np.var(y_v, ddof=0)
            
            # 3. Apply the formula: cov(pred_l, y) / sqrt(var(pred_all) * var(y))
            denominator = np.sqrt(var_full * var_y)
            
            if denominator > 1e-10:
                C = cov_layer_y / denominator
            else:
                C = 0.0
            
            layer_v_contribs.append(C)
            
        contributions[name] = np.mean(layer_v_contribs)
        # print(f"Layer {name} contribution: {contributions[name]:.4f}")

    return contributions, mse, r2