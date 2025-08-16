"""Calibration script stub."""
from __future__ import annotations
import argparse
from pathlib import Path

from src.utils import calib


def main(data_dir: Path, out: Path):
    # Here we would iterate over labeled data and update thresholds.
    print(f"Calibrating with data from {data_dir}, writing {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("configs/thresholds.yaml"))
    args = ap.parse_args()
    main(args.data_dir, args.out)
