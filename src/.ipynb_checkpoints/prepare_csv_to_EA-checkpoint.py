#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import editdistance

# Import CER/WER utilities
sys.path.append(os.path.abspath(".."))
from utils.metrics import CER, WER


def load_ground_truth(gt_path: Path):
    """Reads gt.txt and returns a DataFrame with ImageID and GroundTruth."""
    rows = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                image_id, text = parts
                rows.append((image_id, text))
    return pd.DataFrame(rows, columns=["ImageID", "GroundTruth"])


def compute_metrics(df, wer_mode="tokenizer"):
    """Computes CER and WER for each sample."""
    cer_list, wer_list = [], []

    print("\nComputing CER/WER ...")
    for _, row in df.iterrows():
        gt = str(row["GroundTruth"])
        pred = str(row["Prediction"])

        # CER
        c = CER()
        c.update(pred, gt)
        cer_list.append(c.score())

        # WER
        w = WER(mode=wer_mode)
        w.update(pred, gt)
        wer_list.append(w.score())

    df["CER"] = cer_list
    df["WER"] = wer_list
    return df


def compute_weighted_cer(df):
    """Computes weighted CER for sanity checking."""
    total_dist, total_len = 0, 0
    for _, row in df.iterrows():
        pred = str(row["Prediction"])
        gt = str(row["GroundTruth"])
        total_dist += editdistance.eval(pred, gt)
        total_len += len(gt)
    if total_len == 0:
        return 0.0
    return total_dist / total_len


def main():
    parser = argparse.ArgumentParser(description="Prepare ranked CER file for Stage 2 Human Review.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the model's prediction CSV file (containing GroundTruth & Prediction columns).")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to gt.txt. If not provided, script will infer it from csv_path parent folder.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the ranked CSV output.")
    parser.add_argument("--wer_mode", type=str, default="tokenizer",
                        help="WER mode (default: tokenizer).")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    wer_mode = args.wer_mode

    # Infer gt.txt if not provided
    if args.gt_path is None:
        print("⚠️  No --gt_path provided. Attempting automatic detection ...")
        guess_gt = csv_path.parent.parent / "gt.txt"
        if guess_gt.exists():
            gt_path = guess_gt
            print(f"   → Using inferred gt.txt at: {gt_path}")
        else:
            raise FileNotFoundError("Cannot infer gt.txt path. Please provide --gt_path explicitly.")
    else:
        gt_path = Path(args.gt_path)

    print(f"\nLoading prediction CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Reading ground truth: {gt_path}")
    gt_df = load_ground_truth(gt_path)

    # ----- SAFETY CHECKS -----
    print("\nSanity checks:")
    print("Rows in gt.txt:", len(gt_df))
    print("Rows in CSV:   ", len(df))

    if len(gt_df) != len(df):
        raise ValueError("❌ Row count mismatch — cannot attach ImageIDs by position.")

    mismatch = (
        df["GroundTruth"].astype(str).str.strip()
        != gt_df["GroundTruth"].astype(str).str.strip()
    )
    if mismatch.sum() > 0:
        print("⚠️ GT mismatch detected. Showing first few inconsistencies:")
        print(
            pd.concat(
                [
                    gt_df[mismatch].head(10).add_prefix("gt_"),
                    df[mismatch].head(10).add_prefix("csv_"),
                ],
                axis=1,
            )
        )
        raise ValueError("GT text mismatch — positional mapping unsafe.")

    # ----- ATTACH IMAGEID -----
    df.insert(0, "ImageID", gt_df["ImageID"].values)

    # ----- COMPUTE CER/WER -----
    df = compute_metrics(df, wer_mode)

    # ----- SORT BY CER -----
    df = df.sort_values(by="CER", ascending=False).reset_index(drop=True)

    # ----- SAVE -----
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Ranked CER file saved → {output_path}")

    # ----- WEIGHTED CER -----
    weighted_cer = compute_weighted_cer(df)
    print(f"Weighted mean CER: {weighted_cer:.4f}")


if __name__ == "__main__":
    main()