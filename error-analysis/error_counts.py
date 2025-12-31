#!/usr/bin/env python3
import sys
import pandas as pd

def print_error_counts(csv_path):
    """
    Reads a HITL labels file and prints:
      - Count per error category (Label column)
      - Total samples (excluding empty & 'valid but hard')
    """

    print(f"\nProcessing file: {csv_path}\n")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Fix BOM if present
    df.rename(columns={df.columns[0]: df.columns[0].lstrip("\ufeff")},
              inplace=True)

    # Normalize Label column
    df["Label"] = df["Label"].fillna("").astype(str).str.strip()

    # Exclude:
    #   - empty labels
    #   - 'valid but hard'
    mask_nonempty = df["Label"] != ""
    mask_not_vbh  = df["Label"].str.lower() != "valid but hard"

    df_filtered = df[mask_nonempty & mask_not_vbh].copy()

    # Count per error category (REAL LABEL column)
    counts = df_filtered["Label"].value_counts().sort_index()

    print("Counts per error category (using REAL 'Label' column):\n")
    for label, cnt in counts.items():
        print(f"{label}: {cnt}")

    print(f"\nTotal samples: {len(df_filtered)}\n")
    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python error_counts.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print_error_counts(csv_path)