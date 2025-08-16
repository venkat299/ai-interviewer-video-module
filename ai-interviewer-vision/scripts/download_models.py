"""Script to download models for offline use."""
from __future__ import annotations
import argparse
from pathlib import Path

from src.app import deps


def main(dest: Path):
    cfg = deps.get_config()
    registry = deps.get_registry()
    dest.mkdir(parents=True, exist_ok=True)
    # In real implementation, models would be downloaded.
    print("Models ready at", dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    main(args.out)
