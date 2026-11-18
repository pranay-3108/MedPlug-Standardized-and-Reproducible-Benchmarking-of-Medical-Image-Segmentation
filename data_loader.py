from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import config
from utils import build_mask_from_rles


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
ALIGNED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SIIMDataset(Dataset):
    """SIIM-ACR dataset with backward-compatible DICOM/RLE and aligned-image support.

    source_mode='auto' keeps the old DICOM/RLE path when it exists. If a DICOM
    directory is absent, it uses aligned_dataset/images + aligned_dataset/masks
    when those files exist. The sample choice is recorded in source_by_id.
    """

    def __init__(
        self,
        csv_file=None,
        dicom_dir=None,
        input_channels=None,
        normalization=None,
        image_size=None,
        aligned_image_dir=None,
        aligned_mask_dir=None,
        source_mode="auto",
    ):
        csv_file = Path(csv_file) if csv_file is not None else config.TRAIN_CSV
        dicom_dir = Path(dicom_dir) if dicom_dir is not None else config.DICOM_TRAIN_DIRECTORY

        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        if "ImageId" not in df.columns or "EncodedPixels" not in df.columns:
            raise ValueError(f"Expected ImageId and EncodedPixels columns in {csv_file}")

        self.mask_data = df.groupby("ImageId")["EncodedPixels"].apply(list)
        self.ids = list(self.mask_data.index.astype(str))
        self.csv_path = csv_file
        self.dicom_dir = dicom_dir
        self.aligned_image_dir = Path(aligned_image_dir) if aligned_image_dir else None
        self.aligned_mask_dir = Path(aligned_mask_dir) if aligned_mask_dir else None
        self.source_mode = str(source_mode or "auto").lower()
        if self.source_mode not in {"auto", "dicom", "aligned"}:
            raise ValueError("source_mode must be 'auto', 'dicom', or 'aligned'.")

        self.img_map = {}
        if dicom_dir.is_dir() and self.source_mode != "aligned":
            for img_path in dicom_dir.rglob("*.dcm"):
                self.img_map[img_path.stem] = img_path

        self.aligned_img_map = {}
        if self.aligned_image_dir and self.aligned_image_dir.is_dir() and self.source_mode != "dicom":
            for img_path in self.aligned_image_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in ALIGNED_SUFFIXES:
                    self.aligned_img_map[img_path.stem] = img_path

        self.aligned_mask_map = {}
        if self.aligned_mask_dir and self.aligned_mask_dir.is_dir() and self.source_mode != "dicom":
            for mask_path in self.aligned_mask_dir.rglob("*"):
                if mask_path.is_file() and mask_path.suffix.lower() in ALIGNED_SUFFIXES:
                    self.aligned_mask_map[mask_path.stem] = mask_path

        self.source_by_id = {}
        missing = []
        for img_id in self.ids:
            if self.source_mode in {"auto", "aligned"}:
                if img_id in self.aligned_img_map and img_id in self.aligned_mask_map:
                    self.source_by_id[img_id] = "aligned"
                    continue
            if self.source_mode in {"auto", "dicom"} and img_id in self.img_map:
                self.source_by_id[img_id] = "dicom"
                continue
            missing.append(img_id)

        if missing:
            first = missing[0]
            raise RuntimeError(
                f"Missing image source for {len(missing)} ImageId(s) in {csv_file}. "
                f"First missing ImageId: {first}. "
                f"Available aligned images={len(self.aligned_img_map)}, masks={len(self.aligned_mask_map)}, "
                f"DICOMs={len(self.img_map)}. source_mode={self.source_mode}."
            )

        self.source_counts = {
            "aligned": sum(1 for value in self.source_by_id.values() if value == "aligned"),
            "dicom": sum(1 for value in self.source_by_id.values() if value == "dicom"),
            "total": len(self.source_by_id),
        }

        self.input_channels = input_channels if input_channels is not None else config.INPUT_CHANNELS
        self.normalization = normalization if normalization is not None else config.NORMALIZATION
        self.image_size = tuple(image_size or config.IMAGE_SIZE)

        if self.input_channels not in {1, 3}:
            raise ValueError("INPUT_CHANNELS must be 1 or 3.")
        if self.normalization not in {"minmax", "imagenet"}:
            raise ValueError("NORMALIZATION must be 'minmax' or 'imagenet'.")
        if self.normalization == "imagenet" and self.input_channels != 3:
            raise ValueError("ImageNet normalization requires INPUT_CHANNELS=3.")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _normalize_minmax(img_array):
        img_min = float(img_array.min())
        img_max = float(img_array.max())
        if img_max <= img_min:
            return np.zeros_like(img_array, dtype=np.float32)
        return ((img_array - img_min) / (img_max - img_min)).astype(np.float32)

    def _resize(self, image, mask):
        target_h, target_w = self.image_size
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

        if tuple(image.shape) != (target_h, target_w):
            image_tensor = F.interpolate(
                image_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            mask_tensor = F.interpolate(
                mask_tensor,
                size=(target_h, target_w),
                mode="nearest",
            )

        return image_tensor.squeeze(0).squeeze(0).numpy(), mask_tensor.squeeze(0).squeeze(0).numpy()

    def _build_image_tensor(self, img_array):
        img_array = self._normalize_minmax(img_array)

        if self.input_channels == 1:
            channels = img_array[None, ...]
            return torch.from_numpy(channels.astype(np.float32))

        channels = np.stack([img_array, img_array, img_array], axis=0).astype(np.float32)
        if self.normalization == "imagenet":
            channels = (channels - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        return torch.from_numpy(channels.astype(np.float32))

    @staticmethod
    def _read_aligned(path):
        image = Image.open(path).convert("L")
        return np.asarray(image, dtype=np.float32)

    def __getitem__(self, idx):
        img_id = str(self.ids[idx])
        rles = self.mask_data[img_id]
        source = self.source_by_id[img_id]

        if source == "aligned":
            img_array = self._read_aligned(self.aligned_img_map[img_id])
            mask_array = self._read_aligned(self.aligned_mask_map[img_id])
            mask_array = (mask_array > 127).astype(np.float32)
        else:
            img_path = self.img_map[img_id]
            dcm = pydicom.dcmread(img_path)
            img_array = dcm.pixel_array.astype(np.float32)
            mask_array = build_mask_from_rles(rles, img_array.shape).astype(np.float32)

        img_array, mask_array = self._resize(img_array, mask_array)
        img_tensor = self._build_image_tensor(img_array)
        mask_tensor = torch.from_numpy((mask_array > 0.5)[None, ...].astype(np.float32))

        return img_tensor, mask_tensor
