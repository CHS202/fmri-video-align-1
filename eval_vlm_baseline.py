"""
Zero-shot VLM baselines for the design-classification and experience-prediction
tasks, evaluated against the same train/val splits your brain-aligned models use.

This deliberately reuses your existing `make_dataset` / `get_video_class` loader
so the video list, labels, and ground-truth annotations are identical to what
your co-trained backbones see -- only the video decoding differs (raw video
file fed straight to the VLM, not the pre-extracted frame tensors your
spatial/temporal_transform pipeline builds).

The model is loaded ONCE and reused across every --splits x --runs
combination -- reloading a 7B model per run was the actual bottleneck, not
inference itself. --split/--run (singular) still work as one-item lists for
backward compatibility with old commands.

Usage (one line, all 4 splits x 7 runs = 28 evaluations, one model load):
    python eval_vlm_baseline.py \
        --model llava --task annot-reg \
        --annotation_path video_id_rt_annot.csv \
        --splits 1 2 3 4 --runs 1 2 3 4 5 6 7

    python eval_vlm_baseline.py --model qwen --task design --splits 1
"""
import argparse
import csv
import hashlib
import re
import subprocess
import time
import os
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

# TODO: point this at wherever get_video_class actually lives in your codebase
from models.vlm_backbones import load_scorer
from core.prompts import (
    build_design_prompt,
    build_experience_prompt,
    build_describe_prompt,
    parse_design_response,
    parse_experience_response,
    parse_describe_response,
    DESIGN_CLASSES,
)

RAW_VIDEO_EXT = ".mp4"

# --- media blinding ------------------------------------------------------
# Ground-truth style codes live in the raw filename/directory
# (e.g. "MUJI/SPACE_05_MUJI_clip_000.mp4"), so that path can never be handed
# to the VLM as-is -- even for local models where only pixels should matter,
# "the model probably won't look at the filename" isn't an experimental
# control. Every provider-facing path is instead a content-addressed,
# metadata-stripped copy with an opaque id; the mapping back to the real
# video lives only in a private, 0600 crosswalk file that's never passed to
# scorer.generate().
STYLE_CODE_PATTERN = re.compile("|".join(DESIGN_CLASSES), re.IGNORECASE)

_crosswalk_cache = {}  # crosswalk_path -> set of blind_ids already written, this process


def _assert_path_not_leaking_style(path):
    """Fail closed: raise rather than let a provider-facing path slip through
    with a style code still in it. This should be unreachable if blinding
    ran correctly -- treat a hit here as a bug in get_blinded_video_path(),
    not something to catch and route around."""
    if STYLE_CODE_PATTERN.search(path):
        raise RuntimeError(
            f"Refusing to send a provider-facing path that still contains a "
            f"style code: {path!r}"
        )


def _load_crosswalk_ids(crosswalk_path):
    ids = set()
    if os.path.exists(crosswalk_path):
        with open(crosswalk_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["blind_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def _append_crosswalk_entry(crosswalk_path, blind_id, video_name, raw_path):
    """Appends one blind_id -> original video mapping, deduped against what's
    already on disk (so reruns against an existing --blind_media_dir don't
    pile up duplicate lines) and chmod'd 0600 after every write. This file
    is for the researcher's own bookkeeping only -- never read it back into
    anything that touches the VLM."""
    if crosswalk_path not in _crosswalk_cache:
        _crosswalk_cache[crosswalk_path] = _load_crosswalk_ids(crosswalk_path)
    seen = _crosswalk_cache[crosswalk_path]
    if blind_id in seen:
        return
    entry = {"blind_id": blind_id, "video_name": video_name, "raw_path": raw_path}
    with open(crosswalk_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    os.chmod(crosswalk_path, 0o600)
    seen.add(blind_id)

def get_blinded_video_path(opt, video_name, raw_path):
    """Returns the path that should actually be passed to scorer.generate().

    blind_id is a hash of video_name (not random) so reruns against the same
    --blind_media_dir reuse the already-blinded file instead of re-encoding
    every video on every split/run -- the scorer is loaded once and reused
    across the whole splits x runs loop, and these videos repeat across
    every run, so re-encoding per-call would be wasted ffmpeg work.

    The blinded copy is re-encoded (not stream-copied) with metadata and
    audio stripped, so nothing about the original filename/path can survive
    in the container. Symlinks are deliberately not used, since a provider
    adapter that resolves the real target would re-expose the original path.
    """
    blind_id = hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:16]
    blind_subdir = os.path.join(opt.blind_media_dir, f"media_{blind_id}")
    os.makedirs(blind_subdir, exist_ok=True)
    blind_path = os.path.join(blind_subdir, "clip.mp4")

    if not os.path.exists(blind_path):
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-map_metadata", "-1",
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-movflags", "+faststart",
                blind_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    crosswalk_path = os.path.join(opt.blind_media_dir, "PRIVATE_DO_NOT_SEND_media_crosswalk.jsonl")
    _append_crosswalk_entry(crosswalk_path, blind_id, video_name, raw_path)

    _assert_path_not_leaking_style(blind_path)
    return blind_path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava", "gemini"], required=True)
    p.add_argument("--task", choices=["design", "annot-reg", "describe"], required=True)
    p.add_argument("--dataset_choose", default="rt")
    p.add_argument("--subset", default="validation", choices=["training", "validation"])
    p.add_argument("--splits", type=int, nargs="+", default=[1],
                    help="one or more split indices to run in a single process, "
                         "e.g. --splits 1 2 3 4")
    p.add_argument("--runs", type=int, nargs="+", default=[1],
                    help="one or more run indices per split, e.g. --runs 1 2 3 4 5 6 7 "
                         "(each run is a separate output file -- useful with sampling, "
                         "since do_sample=True means different runs can legitimately "
                         "give different ratings)")
    p.add_argument("--video_path", default="video--imgs",
                    help="root of the pre-extracted frame directories (needed by make_dataset "
                         "for its n_frames existence check, even though the VLM uses --video_raw_path)")
    p.add_argument("--video_raw_path", default="video--raw",
                    help="root of the raw video files actually fed to the VLM")
    p.add_argument("--annotation_path", required=True)
    p.add_argument("--data_root_path", default="BrainGuided")
    p.add_argument("--num_frames", type=int, default=None,
                    help="override the scorer's default frame-sampling count")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5,
                    help="sampling temperature; 0.0 = greedy decoding. Greedy tends to "
                         "collapse ambiguous rating items (e.g. color_comfort, valence) "
                         "to the same value on every video -- see vlm_backbones.py docstring")
    p.add_argument("--top_p", type=float, default=0.9,
                    help="nucleus sampling top_p, only used when --temperature > 0")
    p.add_argument("--limit", type=int, default=None,
                    help="cap number of videos, for a quick smoke test before a full run")
    p.add_argument("--no_blind_media", action="store_true",
                    help="disable filename blinding before the video is sent to the VLM. "
                         "Blinding is ON by default -- raw video paths contain the "
                         "style-code ground truth (MODN/MUJI/SCAN/WABI) in the filename, "
                         "so for any run whose numbers you intend to trust, leave this off. "
                         "Only pass this for local debugging.")
    p.add_argument("--blind_media_dir", default=None,
                    help="where blinded (re-encoded, metadata-stripped) video copies and "
                         "the private crosswalk file are written; defaults to "
                         "<data_root_path>/blinded_media")
    return p.parse_args()


def resolve_data_paths(opt):
    """
    One-time resolution of the local (relative) paths defined in opts.py into
    absolute paths rooted at opt.data_root_path. Deliberately does NOT touch
    opt.output_csv (that depends on opt.split/opt.run, which change every
    iteration of the splits x runs loop) -- see resolve_output_csv() below.
    Calling this more than once would double-join data_root_path, so it's
    only called once in main(), before the loop.
    """
    if opt.data_root_path != '':
        opt.video_path = os.path.join(opt.data_root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.data_root_path, opt.annotation_path)
        opt.video_raw_path = os.path.join(opt.data_root_path, opt.video_raw_path)
        
    opt.blind_media = not opt.no_blind_media
    if opt.blind_media_dir is None:
        opt.blind_media_dir = os.path.join(opt.data_root_path, "blinded_media")
    if opt.blind_media:
        os.makedirs(opt.blind_media_dir, exist_ok=True)
        print(f"Media blinding ON -- provider-facing videos written to {opt.blind_media_dir}")
    else:
        print("Media blinding OFF (--no_blind_media) -- raw filenames, which contain the "
              "style-code ground truth, will be sent straight to the VLM. Don't trust "
              "any resulting numbers.")
    return opt


def resolve_output_csv(opt):
    """Builds the per-split-per-run output path; safe to call every
    iteration since it only reads opt.split/opt.run, never re-joins
    data_root_path onto anything."""
    result_path = os.path.join(opt.data_root_path, f"debug/minor/new_result/{opt.task}/vlm_full-video_blind/{opt.model}/")
    os.makedirs(result_path, exist_ok=True)
    return os.path.join(result_path, f"split-{opt.split}_run-{opt.run}_result.csv")


def get_dimension_names(annotation_path):
    df = pd.read_csv(annotation_path)
    return df.columns[2:].tolist()


def metrics_path(output_csv):
    """Derives the metrics-summary path from the raw-response output_csv path,
    e.g. results/split-1_result.csv -> results/split-1_result_metrics.json,
    so the two files for a given run sit next to each other."""
    base, _ext = os.path.splitext(output_csv)
    return base + "_metrics.json"


def run_design(opt, scorer, video_names, annotations, output_csv):
    prompt = build_design_prompt()
    rows = []
    y_true, y_pred = [], []
    correct_confidences, incorrect_confidences = [], []

    for i, video_name in enumerate(video_names):
        raw_path = os.path.join(opt.video_raw_path, video_name + RAW_VIDEO_EXT)
        gt_label = annotations[i]["label"]

        provider_path = get_blinded_video_path(opt, video_name, raw_path) if opt.blind_media else raw_path
        response = scorer.generate(provider_path, prompt)
        pred_label, confidence, description = parse_design_response(response)

        rows.append({
            "video": video_name, "gt": gt_label, "pred": pred_label,
            "confidence": confidence, "description": description,
            "raw_response": response,
        })
        if pred_label is not None:
            y_true.append(gt_label)
            y_pred.append(pred_label)
            if confidence is not None:
                (correct_confidences if pred_label == gt_label else incorrect_confidences).append(confidence)
        else:
            print(f"[warn] unparseable response for {video_name}: {response[:120]!r}")

        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(video_names)}] running acc so far: "
                  f"{accuracy_score(y_true, y_pred) if y_true else float('nan'):.3f}")

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    n_parsed = len(y_true)
    n_total = len(video_names)
    print(f"\nParsed {n_parsed}/{n_total} responses successfully "
          f"({n_total - n_parsed} failed to parse and were excluded from metrics).")
    if n_parsed == 0:
        print("No parseable predictions -- check prompt/parsing before trusting any metric.")
        return
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=DESIGN_CLASSES, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=DESIGN_CLASSES)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred), order =", DESIGN_CLASSES)
    print(cm)

    # Per-class accuracy (recall): of the videos truly belonging to a class,
    # what fraction got that class as the prediction. Read straight off the
    # confusion matrix diagonal / row sums rather than recomputed from
    # scratch, so it's guaranteed consistent with the printed cm above.
    # n_support can be 0 for a class if every one of its videos failed to
    # parse -- reported as None rather than dividing by zero.
    per_class_accuracy = {}
    y_true_arr = np.array(y_true)
    for row_idx, cls in enumerate(DESIGN_CLASSES):
        n_support = int(cm[row_idx].sum())
        cls_acc = float(cm[row_idx, row_idx] / n_support) if n_support > 0 else None
        per_class_accuracy[cls] = {"accuracy": cls_acc, "n_support": n_support}

    print("\nPer-class accuracy (of ground-truth videos in that class, "
          "fraction correctly predicted):")
    for cls in DESIGN_CLASSES:
        info = per_class_accuracy[cls]
        acc_str = f"{info['accuracy']:.3f}" if info["accuracy"] is not None else "n/a"
        print(f"  {cls:6s}  acc={acc_str}  n={info['n_support']}")

    n_confidence_parsed = len(correct_confidences) + len(incorrect_confidences)
    mean_confidence_correct = float(np.mean(correct_confidences)) if correct_confidences else None
    mean_confidence_incorrect = float(np.mean(incorrect_confidences)) if incorrect_confidences else None
    all_confidences = correct_confidences + incorrect_confidences
    mean_confidence = float(np.mean(all_confidences)) if all_confidences else None

    print(f"\nConfidence: parsed for {n_confidence_parsed}/{n_parsed} predictions")
    if mean_confidence is not None:
        print(f"  mean confidence overall:   {mean_confidence:.4f}")
    if mean_confidence_correct is not None:
        print(f"  mean confidence, correct:   {mean_confidence_correct:.4f}  (n={len(correct_confidences)})")
    if mean_confidence_incorrect is not None:
        print(f"  mean confidence, incorrect: {mean_confidence_incorrect:.4f}  (n={len(incorrect_confidences)})")
    if mean_confidence_correct is not None and mean_confidence_incorrect is not None:
        gap = mean_confidence_correct - mean_confidence_incorrect
        print(f"  calibration gap (correct - incorrect): {gap:+.4f}"
              f"{' -- model is more confident when right, as expected' if gap > 0 else ' -- model shows little/no confidence separation between right and wrong'}")

    summary = {
        "model": opt.model,
        "task": opt.task,
        "split": opt.split,
        "run": opt.run,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix_labels": DESIGN_CLASSES,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
        "n_confidence_parsed": n_confidence_parsed,
        "mean_confidence": mean_confidence,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
    }
    with open(metrics_path(output_csv), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {metrics_path(output_csv)}")

def run_describe(opt, scorer, video_names, annotations, output_csv):
    prompt = build_describe_prompt()
    rows = []
    n_parsed = 0
    n_insufficient = 0
 
    for i, video_name in enumerate(video_names):
        raw_path = os.path.join(opt.video_raw_path, video_name + RAW_VIDEO_EXT)
 
        provider_path = get_blinded_video_path(opt, video_name, raw_path) if opt.blind_media else raw_path
        response = scorer.generate(provider_path, prompt)
        description, insufficient = parse_describe_response(response)
 
        rows.append({
            "video": video_name,
            "description": description,
            "insufficient_visual_evidence": insufficient,
            "raw_response": response,
        })
        if description is not None:
            n_parsed += 1
            if insufficient:
                n_insufficient += 1
        else:
            print(f"[warn] unparseable response for {video_name}: {response[:120]!r}")
 
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(video_names)}] processed, "
                  f"{n_parsed} parsed so far")
 
    pd.DataFrame(rows).to_csv(output_csv, index=False)
 
    n_total = len(video_names)
    print(f"\nParsed {n_parsed}/{n_total} responses successfully "
          f"({n_total - n_parsed} failed to parse).")
    print(f"Flagged insufficient_visual_evidence: {n_insufficient}/{n_parsed if n_parsed else 0}")
 
    summary = {
        "model": opt.model,
        "task": opt.task,
        "split": opt.split,
        "run": opt.run,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_insufficient_visual_evidence": n_insufficient,
    }
    with open(metrics_path(output_csv), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {metrics_path(output_csv)}")

def run_experience(opt, scorer, video_names, annotations, dimension_names, output_csv):
    prompt = build_experience_prompt(dimension_names)
    rows = []
    preds = {dim: [] for dim in dimension_names}
    gts = {dim: [] for dim in dimension_names}

    for i, video_name in enumerate(video_names):
        raw_path = os.path.join(opt.video_raw_path, video_name + RAW_VIDEO_EXT)
        gt_annot = annotations[i]["annot"]  # tensor of length len(dimension_names)
        gt_values = gt_annot.tolist() if hasattr(gt_annot, "tolist") else list(gt_annot)

        provider_path = get_blinded_video_path(opt, video_name, raw_path) if opt.blind_media else raw_path
        response = scorer.generate(provider_path, prompt)
        parsed, description = parse_experience_response(response, dimension_names)

        row = {"video": video_name, "description": description, "raw_response": response}
        for dim, gt_val in zip(dimension_names, gt_values):
            row[f"gt_{dim}"] = gt_val
            pred_val = parsed[dim] if parsed is not None else float("nan")
            row[f"pred_{dim}"] = pred_val
            if not np.isnan(pred_val):
                preds[dim].append(pred_val)
                gts[dim].append(gt_val)
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(video_names)}] processed")

    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print("\nPer-dimension Pearson r (n = number of successfully parsed responses):")
    per_dim = {}
    all_r = []
    for dim in dimension_names:
        n = len(preds[dim])
        if n < 3:
            print(f"  {dim:30s}  n={n}  (too few parsed responses to correlate)")
            per_dim[dim] = {"n": n, "r": None, "p": None, "mae": None}
            continue
        r, p = pearsonr(preds[dim], gts[dim])
        mae = float(np.mean(np.abs(np.array(preds[dim]) - np.array(gts[dim]))))
        all_r.append(r)
        per_dim[dim] = {"n": n, "r": float(r), "p": float(p), "mae": mae}
        print(f"  {dim:30s}  n={n:3d}  r={r:+.3f}  p={p:.3g}  MAE={mae:.3f}")

    mean_r = float(np.mean(all_r)) if all_r else None
    if mean_r is not None:
        print(f"\nMean r across dimensions: {mean_r:+.3f}")

    summary = {
        "model": opt.model,
        "task": opt.task,
        "split": opt.split,
        "run": opt.run,
        "n_total": len(video_names),
        "per_dimension": per_dim,
    }
    with open(metrics_path(output_csv), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {metrics_path(output_csv)}")

def get_video_class_all(opt, annotation_path):
    """Like get_video_class, but ignores train/test splits entirely --
    every row in annotation_path is returned as one big list. No
    train_idx_*.npy / test_idx_*.npy file is loaded, and opt.split /
    subset are not used for selecting rows.
 
    Mirrors the per-task label/annot construction in get_video_class,
    just indexed by row position (i) instead of the 1-based video ids
    pulled out of a split array (i-1).
    """
    df = pd.read_csv(annotation_path)
    video_names = []
    annotations = []
    for i in range(len(df)):
        if opt.task == 'design':
            video_names.append(df.loc[i, 'Video Name and Directory'])  # MODN/SPACE_05_MODN_clip_000
            annotations.append({'label': df.loc[i, 'Video Name and Directory'].split('/')[0], 'annot': torch.tensor([])})  # MODN
        elif opt.task == 'space':
            video_names.append(df.loc[i, 'Video Name and Directory'])
            annotations.append({'label': df.loc[i, 'Video Name and Directory'].split('/')[1][:8], 'annot': torch.tensor([])})  # SPACE_05
        elif opt.task == 'annot':
            video_names.append(df.loc[i, 'Video Name and Directory'])
            annot_value = df.loc[i].iloc[2:].values.tolist()
            thresholds = df.iloc[:, 2:].mean().values
            annot_value_binary = np.where(np.array(annot_value) > thresholds, 1, 0)
            annotations.append({'label': df.loc[i, 'Video Name and Directory'].split('/')[0], 'annot': annot_value_binary})
        elif opt.task == 'annot-reg':
            video_names.append(df.loc[i, 'Video Name and Directory'])
            annot_value = df.loc[i].iloc[2:].values.tolist()
            annotations.append({'label': df.loc[i, 'Video Name and Directory'].split('/')[0], 'annot': torch.tensor(annot_value)})
        elif opt.task == 'describe':
            # no ground truth needed -- free description has nothing to score against
            video_names.append(df.loc[i, 'Video Name and Directory'])
            annotations.append({'label': df.loc[i, 'Video Name and Directory'].split('/')[0], 'annot': torch.tensor([])})
    return video_names, annotations

def main():
    opt = get_args()
    resolve_data_paths(opt)

    # get_video_class() gives us the label/annotation lookup without the
    # frame-existence assertions make_dataset() runs on opt.video_path --
    # since the VLM reads raw video directly, we don't need that check here.
    # from datasets.ve8 import get_video_class  # noqa: E402

    n_total_evals = len(opt.splits) * len(opt.runs)
    print(f"Loading {opt.model} once for {len(opt.splits)} split(s) x {len(opt.runs)} run(s) "
          f"= {n_total_evals} evaluation(s)...")

    scorer_kwargs = {
        "max_new_tokens": opt.max_new_tokens,
        "temperature": opt.temperature,
        "top_p": opt.top_p,
    }
    if opt.num_frames is not None:
        scorer_kwargs["num_frames"] = opt.num_frames

    t_load = time.time()
    scorer = load_scorer(opt.model, **scorer_kwargs)
    print(f"Model loaded in {time.time() - t_load:.1f}s -- reused for all {n_total_evals} run(s) below.")

    dimension_names = get_dimension_names(opt.annotation_path) if opt.task == "annot-reg" else None

    for split in opt.splits:
        opt.split = split
        # video list/labels/annotations depend on split, so this is reloaded
        # per split -- it's cheap (CSV + index lookups), unlike the model.
        video_names, annotations = get_video_class_all(opt, opt.annotation_path)
        if opt.limit:
            video_names = video_names[: opt.limit]
            annotations = annotations[: opt.limit]

        for run in opt.runs:
            opt.run = run
            output_csv = resolve_output_csv(opt)
            print(f"\n=== {opt.model} | split {split} | run {run} -> {output_csv} ===")

            t0 = time.time()
            if opt.task == "design":
                run_design(opt, scorer, video_names, annotations, output_csv)
            elif opt.task == "annot-reg":
                run_experience(opt, scorer, video_names, annotations, dimension_names, output_csv)
            elif opt.task == "describe":
                run_describe(opt, scorer, video_names, annotations, output_csv)
            print(f"split {split} run {run} done in {time.time() - t0:.1f}s")

    print(f"\nAll {n_total_evals} evaluation(s) complete.")


if __name__ == "__main__":
    main()