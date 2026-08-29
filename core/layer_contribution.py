import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
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
        # print(f"Layer {key} - Max value: {np.max(feats_flat)}, Min value: {np.min(feats_flat)}")
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
    # ridge = RidgeCV(alphas=np.logspace(-5, 10, 20))
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_combined, y)
    # print(f"Chosen alpha: {ridge.alpha}")
    
    # ridge.coef_ has shape (N_voxels, Total_PCA_Components)
    all_weights = ridge.coef_

    # --- Calculate TOTAL prediction (all layers combined) ---
    # pred_all shape: (N_samples, N_voxels)
    pred_all = X_combined @ all_weights.T + ridge.intercept_
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

    return pred_all, contributions, mse, r2

def calculate_layer_contributions_v2(all_labels, layers_output, opt):
    """
    per‑layer normalization, preserving within‑layer variances
    """
    
    reduced_features_list = []
    layer_metadata = [] # To keep track of which columns belong to which layer
    current_col_idx = 0

    # --- PCA per layer and prepare for concatenation ---
    # print("Performing PCA and preparing features...")
    for key, feats in layers_output.items():
        feats = feats.detach().cpu().numpy()
        if len(feats.shape) > 2:
            feats_flat = feats.reshape(feats.shape[0], -1)
        else:
            feats_flat = feats

        # PCA: keep 99% variance
        pca = PCA(n_components=0.99)
        feats_pca = pca.fit_transform(feats_flat)  # shape: N_samples x N_components
        print(f"Layer {key} PCA shape: {feats_pca.shape}")

        # NEW: optional per-layer scaling (single scalar per layer)
        # This keeps the relative variance between PCA components within the layer.
        layer_std = feats_pca.std()  # global std for this layer
        if layer_std > 1e-8:
            feats_layer_scaled = feats_pca / layer_std
        else:
            feats_layer_scaled = feats_pca

        n_components = feats_layer_scaled.shape[1]
        reduced_features_list.append(feats_layer_scaled)
        layer_metadata.append({
            "name": key,
            "start": current_col_idx,
            "end": current_col_idx + n_components
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
    # ridge = RidgeCV(alphas=np.logspace(-5, 10, 20))
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_combined, y)
    # print(f"Chosen alpha: {ridge.alpha}")
    
    # ridge.coef_ has shape (N_voxels, Total_PCA_Components)
    all_weights = ridge.coef_

    # --- Calculate TOTAL prediction (all layers combined) ---
    # pred_all shape: (N_samples, N_voxels)
    pred_all = X_combined @ all_weights.T + ridge.intercept_
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

    return pred_all, contributions, mse, r2

def calculate_layer_contributions_v3(train_y, train_feats, val_y, val_feats, opt):
    """
    Fits PCA and Ridge on Train, evaluates on Val.
    """
    reduced_train_list = []
    reduced_val_list = []
    layer_metadata = []
    current_col_idx = 0

    print("Fitting PCA on training data and transforming...")
    
    for key in train_feats.keys():
        t_f = train_feats[key]
        v_f = val_feats[key] # Corresponding validation feature

        # 1. Fit PCA on TRAIN
        pca = PCA(n_components=0.99)
        t_pca = pca.fit_transform(t_f)
        
        # 2. Transform VAL using Train-PCA
        v_pca = pca.transform(v_f)

        # 3. Scaling (using Train stats)
        layer_std = t_pca.std()
        if layer_std > 1e-8:
            t_scaled = t_pca / layer_std
            v_scaled = v_pca / layer_std
        else:
            t_scaled = t_pca
            v_scaled = v_pca

        del t_pca
        del v_pca
        n_components = t_scaled.shape[1]
        
        reduced_train_list.append(t_scaled)
        reduced_val_list.append(v_scaled)
        del t_scaled
        del v_scaled
        
        layer_metadata.append({
            "name": key,
            "start": current_col_idx,
            "end": current_col_idx + n_components
        })
        current_col_idx += n_components

    # Concatenate features
    X_train = np.hstack(reduced_train_list)
    X_val = np.hstack(reduced_val_list)
    del reduced_train_list
    del reduced_val_list
    
    if train_y.ndim == 1: train_y = train_y[:, np.newaxis]
    if val_y.ndim == 1: val_y = val_y[:, np.newaxis]

    # --- Fit Ridge on TRAIN ---
    print(f"Fitting Ridge on Training Data (Shape: {X_train.shape})...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, train_y)
    
    all_weights = ridge.coef_ # (N_voxels, Total_Features)
    intercept = ridge.intercept_

    # --- Evaluate on VALIDATION ---
    print("Evaluating on Validation Data...")
    
    # Total prediction on Val
    pred_val_all = X_val @ all_weights.T + intercept
    
    # Calculate Metrics on Val
    mse = np.mean((pred_val_all - val_y) ** 2, axis=0)
    mse = np.mean(mse) # Avg across voxels
    
    r2 = 1 - (np.mean((pred_val_all - val_y) ** 2, axis=0) / np.var(val_y, axis=0))
    r2 = np.mean(r2) # Avg across voxels

    # --- Calculate Contributions (using Validation Data) ---
    contributions = {}
    
    for meta in layer_metadata:
        name = meta['name']
        start, end = meta['start'], meta['end']
        
        # Layer sub-features (from Val) and weights
        phi_l_val = X_val[:, start:end]
        W_l = all_weights[:, start:end]

        if opt.split == 1:
            df_weights = pd.DataFrame(W_l, 
                                        index=[f"Voxel_{i}" for i in range(W_l.shape[0])],
                                        columns=[f"Comp_{i}" for i in range(W_l.shape[1])])
            df_weights.to_csv(os.path.join(opt.result_path, f"weights_{opt.roi}_{opt.network_choose}_{name}_{opt.data_use}_{opt.split}.csv"))
            del df_weights
        
        # Prediction of just this layer
        pred_l_val = phi_l_val @ W_l.T

        if opt.split == 1:
            # shapes: (n_samples, n_voxels)
            dev_layer = pred_l_val - pred_l_val.mean(axis=0)
            dev_y = val_y - val_y.mean(axis=0)
            
            # 2. Calculate Numerator (Point-wise Co-variance term)
            # This represents how much this specific point matches the target pattern
            # Shape: (n_samples, n_voxels)
            pointwise_cov = dev_layer * dev_y
            print("Pointwise cov shape:", pointwise_cov.shape)
            
            # 3. Calculate Denominator (Scaling Factor)
            # Variances across the whole batch (Scalars per voxel)
            var_full = np.var(pred_val_all, axis=0)
            var_y = np.var(val_y, axis=0)
            denoms = np.sqrt(var_full * var_y)  # Shape: (n_voxels,)
            
            # 4. Handle Division Safely
            valid_mask = denoms > 1e-10
            safe_denoms = np.where(valid_mask, denoms, 1.0) # Avoid div by zero
            
            # 5. Final Point-wise Calculation
            # Broadcast division: (Samples x Voxels) / (Voxels)
            pointwise_contributions = pointwise_cov / safe_denoms
            
            # Zero out invalid columns
            pointwise_contributions[:, ~valid_mask] = 0.0

            df_pw = pd.DataFrame(
                pointwise_contributions,
                index=[f"Sample_{i}" for i in range(pointwise_contributions.shape[0])],
                columns=[f"Voxel_{i}" for i in range(pointwise_contributions.shape[1])]
            )
            # Saves a matrix where Rows=Samples, Columns=Voxels
            df_pw.to_csv(os.path.join(opt.result_path, f"pointwise_contribs_{opt.roi}_{opt.network_choose}_{name}_{opt.data_use}_{opt.split}.csv"))
            del df_pw
            del pointwise_contributions
        
        layer_v_contribs = []
        
        for v in range(val_y.shape[1]):
            y_v = val_y[:, v]
            p_full_v = pred_val_all[:, v]
            p_layer_v = pred_l_val[:, v]
            
            # Covariance Calculation
            cov_layer_y = np.cov(p_layer_v, y_v, ddof=0)[0, 1]
            
            var_full = np.var(p_full_v, ddof=0)
            var_y = np.var(y_v, ddof=0)
            
            denominator = np.sqrt(var_full * var_y)
            
            if denominator > 1e-10:
                C = cov_layer_y / denominator
            else:
                C = 0.0
            
            layer_v_contribs.append(C)
            
        contributions[name] = np.mean(layer_v_contribs)

    return pred_val_all, contributions, mse, r2

import sys
def calculate_layer_correlation(neural_visual, RSA_target, layers_output, epoch, target_labels, roi, opt):
    """
    per‑layer correlation with target
    """
    correlations = {}
    # print("RSA_target shape",RSA_target.shape)
    # print("target_labels", target_labels)
    unique_labels = torch.unique(target_labels, sorted=True)
    for k, v in layers_output.items():
        # get rdm of feats
        if k != 'temporal_attention':
            if opt.network_choose == 'vit_3d':
                v = v.reshape(neural_visual.size(1),neural_visual.size(0),-1).contiguous()
            else:
                v = v.view(neural_visual.size(1), neural_visual.size(0), -1).contiguous()
            v = torch.mean(v, dim=0)

        v1 = v.detach()
        v2 = v1.cuda(0)
        del v
        del v1
        torch.cuda.empty_cache()
        # rdm = compute_label_rdm_from_features(v2, target_labels, unique_labels, opt) # !!!!!!!!!!!!!!!rdm shape = 4*4!!!!!!!!!!!!!!
        # rdm = torch.nn.functional.cosine_similarity(v2.unsqueeze(1), v2.unsqueeze(0), dim=-1)
        rdm = cosine_similarity_matrix(v2)
        del v2
        torch.cuda.empty_cache()
        # print("rdm shape",rdm.shape)
        if epoch == 0:
            plot_layer_rdm_sample(rdm, RSA_target, opt, filename=f'rdm_{k}_{roi}_split{opt.split}.png')
        # print("rdm shape",rdm.shape)
        # print("target shape",RSA_target.shape)
        # sys.exit(0)
        # get correlation between rdm and target upper triangular
        rdm = rdm.cpu().detach().numpy()
        rdm = rdm[np.triu_indices_from(rdm)]
        target = RSA_target.cpu().detach().numpy()
        target = target[np.triu_indices_from(target)]
        correlations[k] = np.corrcoef(rdm, target)[0,1]
    # sys.exit(0)
    del layers_output
    torch.cuda.empty_cache()

    return correlations 

def cosine_similarity_matrix(v2, eps=1e-8):
    v2_norm = v2 / (v2.norm(dim=1, keepdim=True).clamp(min=eps))
    return v2_norm @ v2_norm.t()

def compute_label_rdm_from_features(v2, target, unique_labels, opt):
    """Average CNN features per label, then compute RDM. v2: [batch, features]"""
    num_classes = len(unique_labels)
    feat_dim = v2.shape[1]
    label_means = torch.zeros(num_classes, feat_dim, device=v2.device)
    for idx, lbl in enumerate(unique_labels):
        mask = (target == lbl)
        label_means[idx] = v2[mask].mean(dim=0)
    label_means = label_means - torch.mean(label_means, dim=0)  # center over classes
    label_means = label_means + torch.randn_like(label_means) * 1e-5 # Prevent zero-norm collapse while preserving gradient flow

    norms = label_means.norm(dim=1)
    if (norms < 1e-6).all():
        return torch.zeros(num_classes, num_classes, device=v2.device)

    if opt.rho4rdm:
        label_ranks = torch.argsort(torch.argsort(label_means, dim=1), dim=1).float()
        label_ranks = label_ranks + torch.randn_like(label_ranks) * 1e-8
        rdm = torch.corrcoef(label_ranks)
        rdm = torch.nan_to_num(rdm, nan=0.0)
    else:
        rdm = torch.nn.functional.cosine_similarity(
            label_means.unsqueeze(1), label_means.unsqueeze(0), dim=-1, eps=1e-8
        )
        rdm = torch.nan_to_num(rdm, nan=0.0)
    return rdm

def plot_layer_rdm(rdm, RSA_target, unique_labels, opt, filename):
    import matplotlib.pyplot as plt
    class_to_idx = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    unique_labels = [idx_to_class[i.item()] for i in unique_labels]
    rdm_np = rdm.cpu().detach().numpy()
    RSA_target_np = RSA_target.cpu().detach().numpy()
    save_path = os.path.join(opt.result_path, f"output_and_voxel_select")
    os.makedirs(save_path, exist_ok=True)

    # plot target
    # if not os.path.exists(os.path.join(opt.result_path, f"output_and_voxel_select/{opt.roi}.png")):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=45)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels, rotation=45)
    im = ax.imshow(RSA_target_np, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title(f'{opt.roi}', fontsize=10)
    # set ylim as 0 to 1
    # ax.set_ylim(-1, 1)
    # ax.set_xlabel('Stimulus')
    # ax.set_ylabel('Stimulus')
    
    save_path = os.path.join(opt.result_path, f"output_and_voxel_select/{opt.roi}.png")
    # os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # plot rdm
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels, rotation=45)
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels, rotation=45)
    im = ax.imshow(rdm_np, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title(f'{filename.replace(".png", "")}', fontsize=10)
    # ax.set_ylim(-1, 1)
    # ax.set_xlabel('Stimulus')
    # ax.set_ylabel('Stimulus')
    
    save_path = os.path.join(opt.result_path, f"output_and_voxel_select/{filename}")
    # os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_layer_rdm_sample(rdm, RSA_target, opt, filename):
    import matplotlib.pyplot as plt

    rdm_np = rdm.cpu().detach().numpy()
    RSA_target_np = RSA_target.cpu().detach().numpy()
    n_stimuli = rdm_np.shape[0]

    save_dir = os.path.join(opt.result_path, "output_and_voxel_select")
    os.makedirs(save_dir, exist_ok=True)

    # Target RDM only needs to be plotted once per ROI (it doesn't change across layers/epochs)
    target_path = os.path.join(save_dir, f"{opt.roi}.png")
    if not os.path.exists(target_path):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(RSA_target_np, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{opt.roi} target RDM (n={n_stimuli})', fontsize=10)
        ax.set_xlabel('Stimulus index')
        ax.set_ylabel('Stimulus index')
        fig.savefig(target_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Layer RDM
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rdm_np, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{filename.replace(".png", "")} (n={n_stimuli})', fontsize=10)
    ax.set_xlabel('Stimulus index')
    ax.set_ylabel('Stimulus index')
    layer_path = os.path.join(save_dir, filename)
    fig.savefig(layer_path, dpi=150, bbox_inches='tight')
    plt.close(fig)