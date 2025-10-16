from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from . import dataloader


def _write_list(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for name in names:
            fp.write(f"{name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BCSS-WSSS dataset metadata files.")
    parser.add_argument("--data-root", type=Path, default=Path("BCSS-WSSS"), help="Path to the BCSS-WSSS dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("bcss"), help="Directory to store generated split files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Portion of samples assigned to the training split.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Portion of samples assigned to the validation split. Remaining items go to the test split if any.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffling.")
    parser.add_argument(
        "--write-cls-labels",
        action="store_true",
        help="Persist the computed multi-label annotations to output_dir/cls_labels.npy for faster reloads.",
    )

    args = parser.parse_args()

    mask_dir = args.data_root / dataloader.MASKS_DIR_NAME
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    names = sorted(p.stem for p in mask_dir.glob(f"*{dataloader.MASK_EXT}"))
    if not names:
        raise RuntimeError(f"No mask files found under {mask_dir}")

    random.seed(args.seed)
    random.shuffle(names)

    train_count = int(len(names) * args.train_ratio)
    val_count = int(len(names) * args.val_ratio)

    train_names = sorted(names[:train_count])
    val_names = sorted(names[train_count : train_count + val_count])
    test_names = sorted(names[train_count + val_count :])
    infer_names = sorted(names)

    _write_list(args.output_dir / "train.txt", train_names)
    _write_list(args.output_dir / "val.txt", val_names)
    _write_list(args.output_dir / "infer.txt", infer_names)

    if test_names:
        _write_list(args.output_dir / "test.txt", test_names)

    if args.write_cls_labels:
        cls_labels = dataloader._compute_cls_labels(str(args.data_root))
        cache_path = args.output_dir / "cls_labels.npy"
        np.save(cache_path, cls_labels, allow_pickle=True)
        print(f"Saved class label cache to {cache_path}")

    print(
        "Generated splits with counts: train=%d, val=%d, test=%d" % (len(train_names), len(val_names), len(test_names))
    )


if __name__ == "__main__":
    main()
