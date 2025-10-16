from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    module_path: str
    clip_key: str
    num_classes: int
    ignore_label: int
    default_train_list: str
    default_val_list: str
    default_infer_list: str
    cam_baseline_weights: Optional[str]

    @cached_property
    def module(self):
        return importlib.import_module(self.module_path)


_DATASET_REGISTRY: Dict[str, DatasetInfo] = {}


def _register(dataset_info: DatasetInfo) -> None:
    if dataset_info.name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_info.name}' already registered")
    _DATASET_REGISTRY[dataset_info.name] = dataset_info


# VOC12 dataset registration happens lazily to avoid importing modules at load time.
_register(
    DatasetInfo(
        name="voc12",
        module_path="voc12.dataloader",
        clip_key="voc",
        num_classes=20,
        ignore_label=255,
        default_train_list="voc12/train_aug.txt",
        default_val_list="voc12/val.txt",
        default_infer_list="voc12/train_aug.txt",
        cam_baseline_weights="cam-baseline-voc12/res50_cam.pth",
    )
)

# Placeholder for BCSS-WSSS configuration; this is populated on first import of bcss.dataloader
# to avoid circular imports while keeping the registry definitions in a single place.
def ensure_bcss_registered() -> None:
    if "bcss" in _DATASET_REGISTRY:
        return

    from bcss import dataloader as bcss_dataloader  # local import to avoid heavy dependency during startup

    num_classes = len(bcss_dataloader.CLASS_NAMES)

    _register(
        DatasetInfo(
            name="bcss",
            module_path="bcss.dataloader",
            clip_key="bcss",
            num_classes=num_classes,
            ignore_label=bcss_dataloader.IGNORE_LABEL,
            default_train_list="bcss/train.txt",
            default_val_list="bcss/val.txt",
            default_infer_list="bcss/infer.txt",
            cam_baseline_weights=None,
        )
    )


def get_dataset_info(name: str) -> DatasetInfo:
    if name == "bcss":
        ensure_bcss_registered()

    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {sorted(_DATASET_REGISTRY)}") from exc


def get_dataset_module(name: str):
    info = get_dataset_info(name)
    return info.module


def apply_default_lists(args) -> None:
    """Update dataset list paths with dataset-specific defaults if user kept original VOC defaults."""

    dataset = getattr(args, "dataset", "voc12")
    info = get_dataset_info(dataset)

    # Helper to decide if argument still at VOC default.
    def _needs_override(value: str, default: str) -> bool:
        return value == default

    if _needs_override(args.train_list, "voc12/train_aug.txt"):
        args.train_list = info.default_train_list

    if _needs_override(args.val_list, "voc12/val.txt"):
        args.val_list = info.default_val_list

    if _needs_override(args.infer_list, "voc12/train_aug.txt"):
        args.infer_list = info.default_infer_list
