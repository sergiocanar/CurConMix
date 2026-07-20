"""
Compute mAP / F1 / P / R / Acc / Hit metrics (per-video, macro-averaged) over a
CurConMix predictions CSV, using metric_collator.py + triplets_mapping.py's
component-disentanglement tables (i/v/t/iv/it/ivt), instead of the vendored
ivtmetrics package (which only supports the "ivt" component for MultiBypassT40).

Reports metrics per fold (each fold's videos were predicted by that fold's own
held-out checkpoint, so this is a true per-fold cross-validation breakdown),
plus the fold mean +/- std, and logs everything to a file.

Usage:
    python run_triplet_metrics.py outputs_multibypass/predictions/swin_bas_125_<exp>.csv \
        --dataset-name multibypasst40 --num-classes 85 --mode student \
        --split-selector multibypass-2fold
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from metric_collator import compute_triplet_metrics, format_overall_metrics_ascii
from preprocess import split_selector


def get_fold_video_map(split_name: str) -> dict:
    fold_map = split_selector(split_name)
    result = {}
    for fold, video_list in fold_map.items():
        if video_list and isinstance(video_list[0], int):
            video_list = [f"VID{vid:02d}" for vid in video_list]
        fold_idx = fold - 1 if isinstance(fold, int) else fold
        result[fold_idx] = video_list
    return result


def append_mean_column(table: str) -> str:
    """Append a MEAN column (average across I/V/T/IV/IT/IVT) to each metric row."""
    lines = table.splitlines()
    out = []
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "FINAL":
            out.append(line + f" {'MEAN':>8}")
            continue
        if len(parts) == 7 and parts[0] not in ("=", "-"):
            try:
                values = [float(p) for p in parts[1:]]
            except ValueError:
                out.append(line)
                continue
            mean_val = sum(values) / len(values)
            out.append(line + f" {mean_val:>8.3f}")
        else:
            out.append(line)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_csv")
    parser.add_argument("--dataset-name", default="multibypasst40", choices=["multibypasst40", "cholect50"])
    parser.add_argument("--num-classes", type=int, default=85)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--split-selector", default="multibypass-2fold")
    parser.add_argument("--ignore-null", action="store_true", default=True)
    parser.add_argument("--log-file", default=None, help="Path to write the log; defaults next to the predictions CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    label_cols = [f"tri{i}" for i in range(args.num_classes)]
    pred_cols = [str(i) for i in range(args.num_classes)]

    fold_video_map = get_fold_video_map(args.split_selector)

    log_lines = []

    def emit(text):
        print(text)
        log_lines.append(text)

    fold_maps = {}  # fold_idx -> overall_mAP dict (per component)
    for fold_idx in sorted(fold_video_map):
        fold_videos = set(fold_video_map[fold_idx])
        fold_df = df[df["video"].isin(fold_videos)]
        if fold_df.empty:
            emit(f"[fold{fold_idx + 1}] no matching rows in predictions CSV, skipping")
            continue

        labels = torch.tensor(fold_df[label_cols].values, dtype=torch.long)
        preds = torch.tensor(fold_df[pred_cols].values, dtype=torch.float32)
        video_ids = fold_df["video"].tolist()

        results = compute_triplet_metrics(
            preds=preds,
            labels=labels,
            video_ids=video_ids,
            num_classes=args.num_classes,
            dataset_name=args.dataset_name,
            ignore_null_labels=args.ignore_null,
            f1_thresholds=[0.5],
            f1_topk_values=[5, 10, 20],
            get_per_center=False,
        )
        fold_maps[fold_idx] = results["overall_mAP"]

        emit(append_mean_column(format_overall_metrics_ascii(results, mode=f"{args.mode} fold{fold_idx + 1}")))
        emit("")

    # Fold mean +/- std mAP per component
    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    emit("=" * 102)
    emit(f"[{args.mode}] mAP across folds (mean +/- std)")
    emit("=" * 102)
    header = f"{'':<10} {'I':>8} {'V':>8} {'T':>8} {'IV':>8} {'IT':>8} {'IVT':>8} {'MEAN':>8}"
    emit(header)
    emit("-" * 102)

    per_fold_rows = []
    for fold_idx in sorted(fold_maps):
        row = [fold_maps[fold_idx][c] for c in components]
        per_fold_rows.append(row)
        row_mean = sum(row) / len(row)
        emit(f"{'fold' + str(fold_idx + 1):<10} " + " ".join(f"{v:>8.3f}" for v in row) + f" {row_mean:>8.3f}")

    if per_fold_rows:
        arr = np.array(per_fold_rows)
        mean_row = arr.mean(axis=0)
        std_row = arr.std(axis=0)
        emit("-" * 102)
        emit(f"{'MEAN':<10} " + " ".join(f"{v:>8.3f}" for v in mean_row) + f" {mean_row.mean():>8.3f}")
        emit(f"{'STD':<10} " + " ".join(f"{v:>8.3f}" for v in std_row) + f" {std_row.mean():>8.3f}")
    emit("=" * 102)

    log_file = args.log_file
    if log_file is None:
        base = os.path.splitext(os.path.basename(args.pred_csv))[0]
        log_dir = os.path.join(os.path.dirname(args.pred_csv) or ".", "..", "metrics_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{base}_metrics.log")

    with open(log_file, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nLog written to {log_file}")


if __name__ == "__main__":
    main()
