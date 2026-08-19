"""
Download the PTB-XL dataset from PhysioNet.

PTB-XL is ~21,800 12-lead ECG recordings (10s each) with diagnostic labels
organized in a superclass/subclass hierarchy. This script pulls the full
record set (both 100Hz and 500Hz waveforms) plus the metadata CSVs needed
for labels and train/val/test splits.

Usage:
    python scripts/download_ptbxl.py --output data/ptbxl

Note: originally attempted via wfdb.dl_database(), but that fails against the
live 1.0.3 release for two reasons, so this script downloads the official
project ZIP directly instead:
  1. wfdb's default `records="all"` pulls both the 100Hz (records100/) and
     500Hz (records500/) trees, not just the 100Hz set this project uses --
     harmless here since the full ZIP is only ~1.7GB compressed / ~3GB
     unpacked (smaller than the plan's original 100Hz-only estimate), but
     worth knowing if disk is tighter elsewhere.
  2. PhysioNet's RECORDS file for this dataset is missing a newline exactly
     at the boundary between the records100 and records500 sections, so
     wfdb's line-splitting concatenates the last 100Hz record name with the
     first 500Hz one (e.g. "21837_lrrecords500/00000/00001_hr") and 404s.
Confirmed against the live dataset page at
https://physionet.org/content/ptb-xl/1.0.3/ on 2026-08-18.
"""
import argparse
import os
import zipfile

import requests
from tqdm import tqdm

ZIP_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
MARKER_FILE = "ptbxl_database.csv"


def download_zip(zip_path: str) -> None:
    resp = requests.head(ZIP_URL, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))

    if os.path.exists(zip_path) and os.path.getsize(zip_path) == total_size:
        print(f"Zip already downloaded at {zip_path}, skipping.")
        return

    print(f"Downloading PTB-XL zip ({total_size / 1e9:.2f} GB) to {zip_path} ...")
    with requests.get(ZIP_URL, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: str, output_dir: str) -> None:
    marker_path = os.path.join(output_dir, MARKER_FILE)
    if os.path.exists(marker_path):
        print(f"{marker_path} already exists, skipping extraction.")
        return

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # The official zip wraps everything in one top-level directory; flatten
        # it into output_dir so paths match what preprocess.py expects
        # (e.g. "records100/00000/00001_lr").
        top_level_dirs = {n.split("/", 1)[0] for n in names if "/" in n}
        if len(top_level_dirs) == 1:
            prefix = next(iter(top_level_dirs)) + "/"
        else:
            prefix = ""

        for member in tqdm(names, desc="Extracting"):
            if member.endswith("/"):
                continue
            rel_path = member[len(prefix):] if prefix and member.startswith(prefix) else member
            if not rel_path:
                continue
            dest_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())


def download(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "ptbxl.zip")
    download_zip(zip_path)
    extract_zip(zip_path, output_dir)
    print("Download complete.")
    print("Key files to check:")
    print(f"  {output_dir}/ptbxl_database.csv   -- per-record metadata + labels")
    print(f"  {output_dir}/scp_statements.csv    -- diagnostic code -> superclass mapping")
    print(f"  {output_dir}/records100/           -- 100Hz waveform records")
    print(f"  {output_dir}/records500/           -- 500Hz waveform records (unused by default)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PTB-XL from PhysioNet")
    parser.add_argument("--output", type=str, default="data/ptbxl",
                         help="Directory to download PTB-XL into")
    args = parser.parse_args()
    download(args.output)
