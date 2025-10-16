from __future__ import annotations

import os
from typing import Dict, Iterable, List

import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

from misc import imutils

IMAGES_DIR_NAME = "rgbs_colorNormalized"
MASKS_DIR_NAME = "masks"
IMAGE_EXT = ".png"
MASK_EXT = ".png"
IGNORE_LABEL = 255

CLASS_NAMES: List[str] = [
    "tumor",
    "stroma",
    "lymphocytic infiltrate",
    "necrosis or debris",
    "glandular secretions",
    "blood",
    "exclude",
    "metaplasia",
    "fat",
    "plasma cells",
    "other immune infiltrate",
    "mucoid material",
    "normal acinus or duct",
    "lymphatics",
    "undetermined",
    "nerve",
    "skin adnexa",
    "blood vessel",
    "angioinvasion",
    "dcis",
    "other",
]

CLASS_COLORS = None  # placeholder for future customization

N_CAT = len(CLASS_NAMES)

_CAT_NUM_MAP = {idx + 1: idx for idx in range(N_CAT)}

_cls_labels_cache: Dict[str, Dict[str, np.ndarray]] = {}


def decode_int_filename(name: str) -> str:
    """Compatibility wrapper used across the training pipeline."""
    return str(name)


def _mask_path(data_root: str, image_stem: str) -> str:
    return os.path.join(data_root, MASKS_DIR_NAME, f"{image_stem}{MASK_EXT}")


def _image_path(data_root: str, image_stem: str) -> str:
    return os.path.join(data_root, IMAGES_DIR_NAME, f"{image_stem}{IMAGE_EXT}")


def load_img_name_list(dataset_path: str) -> List[str]:
    with open(dataset_path, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp if line.strip()]
    return lines


def _compute_cls_labels(data_root: str) -> Dict[str, np.ndarray]:
    mask_dir = os.path.join(data_root, MASKS_DIR_NAME)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    cls_labels: Dict[str, np.ndarray] = {}

    for fname in os.listdir(mask_dir):
        if not fname.endswith(MASK_EXT):
            continue
        stem = fname[: -len(MASK_EXT)]
        mask = imageio.imread(os.path.join(mask_dir, fname))
        if mask.ndim == 3:
            mask = mask[..., 0]

        label_vec = np.zeros(N_CAT, dtype=np.float32)
        for value in np.unique(mask):
            if value == 0:
                continue  # outside ROI treated as background/ignore
            mapped = _CAT_NUM_MAP.get(int(value))
            if mapped is not None:
                label_vec[mapped] = 1.0
        cls_labels[stem] = label_vec

    return cls_labels


def _get_cls_labels_dict(data_root: str) -> Dict[str, np.ndarray]:
    key = os.path.abspath(data_root)
    if key not in _cls_labels_cache:
        _cls_labels_cache[key] = _compute_cls_labels(data_root)
    return _cls_labels_cache[key]


def load_image_label_list(img_name_list: Iterable[str], data_root: str) -> np.ndarray:
    cls_labels_dict = _get_cls_labels_dict(data_root)

    labels = []
    for name in img_name_list:
        if name not in cls_labels_dict:
            raise KeyError(
                f"Mask for image '{name}' not found while building class labels."
                f" Expected file: {_mask_path(data_root, name)}"
            )
        labels.append(cls_labels_dict[name])
    return np.stack(labels, axis=0)


class TorchvisionNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255.0 - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255.0 - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255.0 - self.mean[2]) / self.std[2]

        return proc_img


class GetAffinityLabelFromIndices:
    def __init__(self, indices_from, indices_to):
        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):
        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(
            np.less(segm_label_from, N_CAT + 1), np.less(segm_label_to, N_CAT + 1)
        )
        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(
            np.float32
        )
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(
            np.float32
        )

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return (
            torch.from_numpy(bg_pos_affinity_label),
            torch.from_numpy(fg_pos_affinity_label),
            torch.from_numpy(neg_affinity_label),
        )


class BCSSImageDataset(Dataset):
    def __init__(
        self,
        img_name_list_path: str,
        data_root: str,
        resize_long=None,
        rescale=None,
        img_normal: TorchvisionNormalize | None = TorchvisionNormalize(),
        hor_flip: bool = False,
        crop_size=None,
        crop_method: str | None = None,
        to_torch: bool = True,
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.data_root = data_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = np.asarray(imageio.imread(_image_path(self.data_root, name)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {"name": name, "img": img}


class BCSSClassificationDataset(BCSSImageDataset):
    def __init__(
        self,
        img_name_list_path: str,
        data_root: str,
        resize_long=None,
        rescale=None,
        img_normal: TorchvisionNormalize | None = TorchvisionNormalize(),
        hor_flip: bool = False,
        crop_size=None,
        crop_method: str | None = None,
    ):
        super().__init__(
            img_name_list_path,
            data_root,
            resize_long=resize_long,
            rescale=rescale,
            img_normal=img_normal,
            hor_flip=hor_flip,
            crop_size=crop_size,
            crop_method=crop_method,
        )
        self.label_list = load_image_label_list(self.img_name_list, data_root)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["label"] = torch.from_numpy(self.label_list[idx])
        return out


class BCSSClassificationDatasetMSF(BCSSClassificationDataset):
    def __init__(self, img_name_list_path, data_root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        self.scales = scales
        super().__init__(img_name_list_path, data_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = imageio.imread(_image_path(self.data_root, name))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {
            "name": name,
            "img": ms_img_list,
            "size": (img.shape[0], img.shape[1]),
            "label": torch.from_numpy(self.label_list[idx]),
        }
        return out


class BCSSSegmentationDataset(Dataset):
    def __init__(
        self,
        img_name_list_path,
        label_dir,
        crop_size,
        data_root,
        rescale=None,
        img_normal=TorchvisionNormalize(),
        hor_flip=False,
        crop_method="random",
    ):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.data_root = data_root
        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

        self.cls_label_list = load_image_label_list(self.img_name_list, data_root)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = imageio.imread(_image_path(self.data_root, name))
        label_path = os.path.join(self.label_dir, f"{name}{MASK_EXT}")
        label = imageio.imread(label_path)

        if label.ndim == 3:
            label = label[..., 0]

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, IGNORE_LABEL))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, IGNORE_LABEL)

        img = imutils.HWC_to_CHW(img)

        return {
            "name": name,
            "img": img,
            "label": label,
            "cls_label": torch.from_numpy(self.cls_label_list[idx]),
        }


class BCSSAffinityDataset(BCSSSegmentationDataset):
    def __init__(
        self,
        img_name_list_path,
        label_dir,
        crop_size,
        data_root,
        indices_from,
        indices_to,
        rescale=None,
        img_normal=TorchvisionNormalize(),
        hor_flip=False,
        crop_method=None,
    ):
        super().__init__(
            img_name_list_path,
            label_dir,
            crop_size,
            data_root,
            rescale=rescale,
            img_normal=img_normal,
            hor_flip=hor_flip,
            crop_method=crop_method,
        )
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        reduced_label = imutils.pil_rescale(out["label"], 0.25, 0)
        (
            out["aff_bg_pos_label"],
            out["aff_fg_pos_label"],
            out["aff_neg_label"],
        ) = self.extract_aff_lab_func(reduced_label)
        return out


class BCSSDatasetForEval(Dataset):
    def __init__(self, img_name_list_path, data_root):
        self.ids = load_img_name_list(img_name_list_path)
        self.data_root = data_root

    def read_label(self, file, dtype=np.int32):
        with Image.open(file) as fp:
            img = fp.convert("P")
            img = np.array(img, dtype=dtype)

        if img.ndim == 2:
            return img
        if img.shape[2] == 1:
            return img[:, :, 0]
        return img

    def get_label(self, name):
        label_path = _mask_path(self.data_root, name)
        label = self.read_label(label_path, dtype=np.int32)
        label[label == IGNORE_LABEL] = -1
        return label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return idx


# Aliases to align with the original VOC dataloader API
ClassificationDataset = BCSSClassificationDataset
ClassificationDatasetMSF = BCSSClassificationDatasetMSF
ImageDataset = BCSSImageDataset
SegmentationDataset = BCSSSegmentationDataset
AffinityDataset = BCSSAffinityDataset
DatasetForEval = BCSSDatasetForEval
