"""
Preprocess PTB-XL: load raw waveforms, map diagnostic codes to the 5
superclasses (NORM, MI, STTC, CD, HYP), filter to single-label records for
a clean baseline, and write a train/val/test split as .npy files.

Usage:
    python scripts/preprocess.py --input data/ptbxl --output data/processed
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_raw_signals(df: pd.DataFrame, records_path: str, sampling_rate: int = 100) -> np.ndarray:
    """Load waveform data for every record referenced in df."""
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signals = []
    for f in tqdm(df[col], desc="Loading waveforms"):
        record = wfdb.rdrecord(os.path.join(records_path, f))
        signals.append(record.p_signal)  # shape: (time, leads)
    return np.array(signals, dtype=np.float32)


def aggregate_diagnostic(scp_codes: dict, agg_df: pd.DataFrame) -> list:
    """Map a record's SCP diagnostic codes to superclass labels."""
    classes = set()
    for code in scp_codes.keys():
        if code in agg_df.index:
            classes.add(agg_df.loc[code, "diagnostic_class"])
    return list(classes)


def main(input_dir: str, output_dir: str, sampling_rate: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(input_dir, "ptbxl_database.csv")
    scp_path = os.path.join(input_dir, "scp_statements.csv")

    df = pd.read_csv(meta_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    print("Mapping SCP codes to superclasses...")
    df["diagnostic_superclass"] = df.scp_codes.apply(
        lambda codes: aggregate_diagnostic(codes, agg_df)
    )

    # Keep single-label records only, for a clean first-pass baseline.
    df = df[df.diagnostic_superclass.apply(len) == 1].copy()
    df["label"] = df.diagnostic_superclass.apply(lambda x: x[0])
    df = df[df.label.isin(SUPERCLASSES)]

    print(f"Records after filtering to single-label superclasses: {len(df)}")
    print(df.label.value_counts())

    # PTB-XL provides a recommended 10-fold split in strat_fold; use fold 10
    # as test, fold 9 as val, rest as train (standard convention in the literature).
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]

    label_to_idx = {c: i for i, c in enumerate(SUPERCLASSES)}

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\nProcessing {name} split ({len(split_df)} records)...")
        X = load_raw_signals(split_df, input_dir, sampling_rate)
        y = split_df.label.map(label_to_idx).values.astype(np.int64)
        np.save(os.path.join(output_dir, f"X_{name}.npy"), X)
        np.save(os.path.join(output_dir, f"y_{name}.npy"), y)
        print(f"  saved X_{name}.npy {X.shape}, y_{name}.npy {y.shape}")

    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        f.write("\n".join(SUPERCLASSES))

    print("\nDone. Label mapping:", label_to_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL for baseline classification")
    parser.add_argument("--input", type=str, default="data/ptbxl")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    args = parser.parse_args()
    main(args.input, args.output, args.sampling_rate)
