"""
python eval_cotrain.py \
    --model video_swin \
    --task annot-reg \
    --approach cotrain \
    --splits 1 2 3 4 \
    --subs 01 02 03 06 08 09 12
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Imports from your codebase modules
from datasets.ve8 import VE8Dataset
from eval_vlm_baseline import get_dimension_names
from transforms.spatial import Scale, CenterCornerCrop, Compose, ToTensor
from transforms.temporal import TSN
from transforms.target import ClassLabel
from models.visual_stream import CNN_3D, VisualStream

DESIGN_CLASSES = ["MODN", "MUJI", "SCAN", "WABI"]
SUB_LIST = ["01", "02", "03", "06", "08", "09", "12"]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, 
                   choices=["resnet_18", "shufflenet_v1", "shufflenet_v2", "squeezenet", 
                            "mobilenet_v1", "mobilenet_v2", "alexnet_3d", 
                            "vit_3d", "video_swin"])
    p.add_argument("--task", type=str, required=True, choices=["design", "annot-reg"])
    p.add_argument("--approach", type=str, required=True, choices=["cotrain", "not-cotrain", "pretrain"])
    p.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--subs", type=str, nargs="+", default=SUB_LIST)
    p.add_argument("--dataset_choose", default="rt")
    p.add_argument("--data_root_path", default="BrainGuided")
    p.add_argument("--video_path", default="RT--imgs")
    p.add_argument("--video_raw_path", default="RT--raw")
    p.add_argument("--sample_size", type=int, default=112)
    p.add_argument("--snippet_duration", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def resolve_paths(opt):
    if opt.data_root_path != "":
        opt.video_path = os.path.join(opt.data_root_path, opt.video_path)
        opt.video_raw_path = os.path.join(opt.data_root_path, opt.video_raw_path)
        
    if opt.task == "annot-reg":
        opt.annotation_path = os.path.join(opt.data_root_path, "video_id_rt_annot.csv")
    else:
        opt.annotation_path = os.path.join(opt.data_root_path, "video_id_rt.csv")
    return opt

def build_dataloader(opt):
    # Match the validation/testing pipeline setup
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCornerCrop(opt.sample_size, crop_position='c'),
        ToTensor(norm_value=1)
    ])
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()

    dataset = VE8Dataset(
        opt=opt,
        video_path=opt.video_path,
        annotation_path=opt.annotation_path,
        subset="validation",
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        dataset_choose=opt.dataset_choose
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True
    )
    return loader

def run_design_eval(model, dataloader, device):
    model.eval()
    all_preds, all_gts, all_vids, all_probs = [], [], [], []
    
    with torch.no_grad():
        for snippets, target, vis_item, _ in dataloader:
            snippets = snippets.to(device)
            output, _, _, _, _ = model(snippets)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_gts.extend(target.numpy())
            all_vids.extend([v[0] if isinstance(v, list) else v for v in vis_item])
            all_probs.append(probs)
            
    return all_vids, all_gts, all_preds, np.vstack(all_probs)

def run_experience_eval(model, dataloader, device):
    model.eval()
    all_preds, all_gts, all_vids = [], [], []
    
    with torch.no_grad():
        for snippets, _, vis_item, annot in dataloader:
            snippets = snippets.to(device)
            # Output represents raw continuous predictions (1-7 or 1-5)
            output, _, _, _, _ = model(snippets)
            
            all_preds.append(output.cpu().numpy())
            all_gts.append(annot.numpy())
            all_vids.extend([v[0] if isinstance(v, list) else v for v in vis_item])
            
    return all_vids, np.vstack(all_gts), np.vstack(all_preds)

def main():
    opt = get_args()
    resolve_paths(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dimension_names = get_dimension_names(opt.annotation_path) if opt.task == "annot-reg" else None
    n_classes = len(dimension_names) if opt.task == "annot-reg" else 4

    if opt.model == "shufflenet_v1":
        opt.model_pretrained = "pretrained-models/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth"
    elif opt.model == 'resnet_18':
        opt.model_pretrained = 'pretrained-models/resnet-18-kinetics.pth'
    else:
        opt.model_pretrained = getattr(opt, "model_pretrained", None)

    ckpt_model_dir = "shufflenet_v1_1.5x" if opt.model == "shufflenet_v1" else opt.model

    for split in opt.splits:
        opt.split = split
        val_loader = build_dataloader(opt)

        for sub in opt.subs:
            # Reconstruct exact checkpoint path from training layout
            if opt.approach == 'cotrain':
                ckpt_path = os.path.join(
                    opt.data_root_path,
                    f"debug/minor/new_result/{opt.task}/sig_test_each_roi_neurostorm_win4/rt/{ckpt_model_dir}/"
                    f"result_rt_split={opt.split}_co_train_alpha=5_use_model=1_lr=0.0002_99_sub-{sub}/checkpoints/best.pth"
                )
            elif opt.approach == 'not-cotrain':
                ckpt_path = os.path.join(
                    opt.data_root_path,
                    f"debug/minor/new_result/{opt.task}/sig_test/rt/{ckpt_model_dir}/"
                    f"result_rt_split={opt.split}_not_co_train_lr=0.0002_{sub}_/checkpoints/best.pth"
                )
            elif opt.approach == 'pretrain':
                ckpt_path = ""
            
            if not os.path.exists(ckpt_path) and opt.approach != 'pretrain':
                print(f"[Skip] Checkpoint not found: {ckpt_path}")
                continue

            print(f"\n=== Evaluating | Model: {opt.model} | Split: {split} | Subject: sub-{sub} ===")

            if opt.model == "resnet_18":
                model = VisualStream(
                    snippet_duration=opt.snippet_duration,
                    sample_size=opt.sample_size,
                    n_classes=n_classes,
                    seq_len=opt.seq_len,
                    pretrained_model_path=opt.model_pretrained,
                ).to(device)
            else:
                model = CNN_3D(
                    snippet_duration=opt.snippet_duration,
                    sample_size=opt.sample_size,
                    n_classes=n_classes,
                    seq_len=opt.seq_len,
                    pretrained_model_path=opt.model_pretrained,
                    network_choose=opt.model
                ).to(device)
            if opt.approach != 'pretrain':
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, nn.Module):
                    state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("state_dict", checkpoint)
                else:
                    state_dict = checkpoint
                
                # Strip potential DataParallel module prefix
                clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(clean_state_dict, strict=True)
                print(f"Loaded checkpoint from: {ckpt_path}")

            if opt.approach == 'cotrain':
                out_dir = os.path.join(opt.data_root_path, f"debug/minor/new_result/{opt.task}/co_train_eval/{opt.model}/")
            elif opt.approach == 'not-cotrain':
                out_dir = os.path.join(opt.data_root_path, f"debug/minor/new_result/{opt.task}/not_cotrain_eval/{opt.model}/")
            elif opt.approach == 'pretrain':
                out_dir = os.path.join(opt.data_root_path, f"debug/minor/new_result/{opt.task}/pretrain_eval/{opt.model}/")
            
            os.makedirs(out_dir, exist_ok=True)
            if opt.approach == 'cotrain':
                output_csv = os.path.join(out_dir, f"split-{split}_sub-{sub}_result.csv")
            elif opt.approach == 'not-cotrain' or opt.approach == 'pretrain':
                output_csv = os.path.join(out_dir, f"split-{split}_run-{sub}_result.csv")

            if opt.task == "design":
                vids, gts, preds, probs = run_design_eval(model, val_loader, device)
                idx_to_class = {i: cls for i, cls in enumerate(DESIGN_CLASSES)}
                
                rows = []
                for i, vid in enumerate(vids):
                    row = {
                        "video": vid,
                        "gt": idx_to_class[gts[i]],
                        "pred": idx_to_class[preds[i]]
                    }
                    for c_idx, cls in idx_to_class.items():
                        row[f"prob_{cls}"] = probs[i, c_idx]
                    rows.append(row)
                pd.DataFrame(rows).to_csv(output_csv, index=False)
                
                acc = accuracy_score(gts, preds)
                print(f"Design Task Accuracy: {acc:.4f}")

            elif opt.task == "annot-reg":
                vids, gts, preds = run_experience_eval(model, val_loader, device)
                
                rows = []
                for i, vid in enumerate(vids):
                    row = {"video": vid}
                    for d_idx, dim in enumerate(dimension_names):
                        # Save raw ground truth and predicted continuous values directly
                        row[f"gt_{dim}"] = gts[i, d_idx]
                        row[f"pred_{dim}"] = preds[i, d_idx]
                    rows.append(row)
                
                pd.DataFrame(rows).to_csv(output_csv, index=False)
                
                print("\nPer-dimension Pearson r & MAE:")
                for d_idx, dim in enumerate(dimension_names):
                    r_val, _ = pearsonr(preds[:, d_idx], gts[:, d_idx])
                    mae = np.mean(np.abs(preds[:, d_idx] - gts[:, d_idx]))
                    print(f"  {dim:30s} | r = {r_val:+.3f} | MAE = {mae:.3f}")

            print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    main()