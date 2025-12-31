#!/usr/bin/env python
import sys
import argparse
import pandas as pd
import editdistance

VBH_LABEL = "Valid but hard"   # special label to always keep

# Group these 3 raw labels into one category "Irrelevant Content"
IRRELEVANT_GROUP_LABEL = "Irrelevant Content"
IRRELEVANT_RAW_SET = {
    "irrelevant / signature",
    "irrelevant / non-text",
    "irrelevant / page number / marginal",
}


def normalize_label(lab: str) -> str:
    """Normalize labels:
       - unify 'Valid but hard'
       - group several irrelevant-* labels into 'Irrelevant Content'
       - strip whitespace, keep empty as ''
    """
    if not isinstance(lab, str):
        return ""

    s = lab.strip()
    if not s:
        return ""

    low = s.lower()

    # Canonical VBH label
    if low == VBH_LABEL.strip().lower():
        return VBH_LABEL

    # Group irrelevant-* variants
    if low in IRRELEVANT_RAW_SET:
        return IRRELEVANT_GROUP_LABEL

    # Otherwise keep as-is (stripped)
    return s


def compute_metrics(df_subset):
    """Compute weighted CER and WER over a subset dataframe."""
    # CER
    dist, total = 0, 0
    for _, r in df_subset.iterrows():
        g, p = str(r["GroundTruth"]), str(r["Prediction"])
        dist += editdistance.eval(p, g)
        total += len(g)
    cer = dist / total if total > 0 else float("nan")

    # WER
    wer_dist, wer_total = 0, 0
    for _, r in df_subset.iterrows():
        g_toks = str(r["GroundTruth"]).split()
        p_toks = str(r["Prediction"]).split()
        wer_dist += editdistance.eval(p_toks, g_toks)
        wer_total += len(g_toks)
    wer = wer_dist / wer_total if wer_total > 0 else float("nan")

    return cer, wer


def main(pred_csv: str, hitl_log: str):
    print(f"Using prediction file: {pred_csv}")
    print(f"Using HITL log file:   {hitl_log}\n")

    # -------- Load predictions --------
    df = pd.read_csv(pred_csv)
    # Fix BOM on first column if present
    df.rename(columns={df.columns[0]: df.columns[0].lstrip("\ufeff")}, inplace=True)

    for col in ("ImageID", "GroundTruth", "Prediction"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {pred_csv}")

    df["ImageID"] = df["ImageID"].astype(str).str.strip()

    # -------- Load HITL log --------
    log_df = pd.read_csv(hitl_log)
    log_df.rename(columns={log_df.columns[0]: log_df.columns[0].lstrip("\ufeff")}, inplace=True)

    if "ImageID" not in log_df.columns or "Label" not in log_df.columns:
        raise ValueError(f"HITL log must contain 'ImageID' and 'Label' columns: {hitl_log}")

    log_df["ImageID"] = log_df["ImageID"].astype(str).str.strip()
    log_df["Label"]   = log_df["Label"].astype(str).str.strip()

    # If multiple entries per ImageID, keep the LAST (by timestamp if available)
    if "Timestamp" in log_df.columns:
        log_df = (
            log_df
            .sort_values("Timestamp")
            .groupby("ImageID", as_index=False)
            .tail(1)
        )
    else:
        log_df = log_df.groupby("ImageID", as_index=False).tail(1)

    # Map ImageID → Label and attach to df
    label_map = dict(zip(log_df["ImageID"], log_df["Label"]))
    df["Label"] = df["ImageID"].map(label_map).fillna("")

    # -------- Normalize labels (group irrelevant variants, canonicalize VBH) --------
    df["Label"] = df["Label"].apply(normalize_label)

    # ======================================================
    #  AUTO-DETECT ERROR CATEGORIES FROM Label COLUMN
    #  (exclude empty labels and "Valid but hard")
    # ======================================================
    labels_series = df["Label"].astype(str).str.strip()
    vbh_norm = VBH_LABEL.strip().lower()

    error_categories = sorted(
        {
            lab
            for lab in labels_series.unique()
            if lab != "" and lab.strip().lower() != vbh_norm
        }
    )

    print("Detected error categories from HITL log (excluding 'Valid but hard'):")
    for lab in error_categories:
        print(" -", lab)
    print()

    # ======================================================
    # 0️⃣ NO CLEANING: use ALL samples
    # ======================================================
    all_cer, all_wer = compute_metrics(df)

    print("🟣 NO CLEANING (all samples kept):")
    print(f"   CER = {all_cer:.4f}, WER = {all_wer:.4f}")
    print(f"   Samples used: {len(df)}\n")

    # ======================================================
    # 1️⃣ CLEAN ALL: keep only Valid-but-hard + UNLABELED
    # ======================================================
    baseline_df = df[
        (df["Label"].str.strip().str.lower() == vbh_norm) |
        (df["Label"] == "")
    ]
    baseline_cer, baseline_wer = compute_metrics(baseline_df)

    print("🔵 Clean all (keep only 'Valid but hard' + unlabeled):")
    print(f"   CER = {baseline_cer:.4f}, WER = {baseline_wer:.4f}")
    print(f"   Samples used: {len(baseline_df)}\n")

    # ======================================================
    # 2️⃣ FOR EACH ERROR CATEGORY:
    #    REMOVE this category, keep everything else
    #    (VBH + unlabeled always kept because they are not in error_categories)
    # ======================================================
    removed_results = []

    for cat in error_categories:
        # Remove all rows whose Label is exactly this category
        df_subset = df[df["Label"] != cat]

        cer, wer = compute_metrics(df_subset)
        removed_results.append(
            (
                cat,
                cer,
                wer,
                len(df_subset),
                cer - all_cer,  # negative => CER improved vs no cleaning
                wer - all_wer,  # negative => WER improved vs no cleaning
            )
        )

    # ------------------------------------------------------
    # Make results table, sorted by CER ASC (best first)
    # ------------------------------------------------------
    if removed_results:
        results_df = pd.DataFrame(
            removed_results,
            columns=[
                "Removed category",
                "CER",
                "WER",
                "Samples used",
                "CER_diff_vs_all",
                "WER_diff_vs_all",
            ],
        )

        results_df = results_df.sort_values(by="CER", ascending=True).reset_index(drop=True)

        print("\n📉 CER/WER when REMOVING each error category "
              "(sorted by CER ASC, lower is better):\n")
        print(results_df.to_string(index=False))
    else:
        print("No error categories found (only 'Valid but hard' and/or unlabeled samples).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze effect of different HITL error categories on CER/WER."
    )
    parser.add_argument(
        "pred_csv",
        help="Path to predictions CSV (with ImageID, GroundTruth, Prediction columns).",
    )
    parser.add_argument(
        "hitl_log",
        help="Path to HITL labels log CSV (with ImageID, Label columns).",
    )
    args = parser.parse_args()

    main(args.pred_csv, args.hitl_log)