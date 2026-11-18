"""Validated SIIM-ACR dataset contract used before training.

Supports the original DICOM/RLE layout and the existing aligned_dataset copy.
The old DICOM path remains preferred when it exists; aligned files are used only
when configured/available for a sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from utils import build_mask_from_rles

EXPECTED_SPLITS = {
    "TRAIN": {"total": 10675, "positive": 2379, "negative": 8296},
    "VALIDATION": {"total": 854, "positive": 172, "negative": 682},
    "TEST": {"total": 854, "positive": 174, "negative": 680},
}
REQUIRED_COLUMNS = {"ImageId", "EncodedPixels"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_dicom_index(dicom_dir: Path, allow_missing: bool = False) -> dict[str, Path]:
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_dir}")
    return {path.stem: path for path in dicom_dir.rglob("*.dcm")}


def build_image_index(image_dir: Path | None) -> dict[str, Path]:
    if not image_dir:
        return {}
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def read_split(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {missing}")
    return df


def split_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)
    positive = sum(
        any(str(rle).strip() not in {"", "-1", "nan"} for rle in rles)
        for rles in grouped
    )
    total = len(grouped)
    return total, positive, total - positive


def _source_maps(dicom_dir, aligned_images, aligned_masks):
    dicom = build_dicom_index(dicom_dir, allow_missing=True)
    aligned_image_map = build_image_index(aligned_images)
    aligned_mask_map = build_image_index(aligned_masks)
    return dicom, aligned_image_map, aligned_mask_map


def _choose_source(image_id: str, dicom, aligned_images, aligned_masks, source_mode: str):
    mode = str(source_mode or "auto").lower()
    if mode not in {"auto", "dicom", "aligned"}:
        raise ValueError(f"Unsupported dataset source_mode: {source_mode}")
    if mode in {"auto", "aligned"} and image_id in aligned_images and image_id in aligned_masks:
        return "aligned"
    if mode in {"auto", "dicom"} and image_id in dicom:
        return "dicom"
    return None


def _check_aligned_sample(image_path: Path, mask_path: Path, expected_positive: bool):
    try:
        image = np.asarray(Image.open(image_path).convert("L"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
    except Exception as exc:
        raise RuntimeError(f"Aligned image/mask could not be read: {image_path}; {mask_path}; error={exc}") from exc
    if image.size == 0 or mask.size == 0:
        raise RuntimeError(f"Aligned image/mask is empty: {image_path}; {mask_path}")
    mask_positive = bool((mask > 127).any())
    if expected_positive and not mask_positive:
        raise RuntimeError(f"Positive aligned mask is empty: {mask_path}")
    if not expected_positive and mask_positive:
        raise RuntimeError(f"Negative aligned mask contains positive pixels: {mask_path}")
    return image.shape, int((mask > 127).sum())


def validate_contract(splits: list[tuple], source_mode: str = "auto", strict_counts: bool = True) -> dict[str, dict]:
    """Validate split counts and source mapping.

    Backward-compatible tuple format:
        (name, csv_path, dicom_dir)

    Extended format:
        (name, csv_path, dicom_dir, aligned_images_dir, aligned_masks_dir)
    """
    results = {}

    for item in splits:
        if len(item) == 3:
            name, csv_path, dicom_dir = item
            aligned_images = None
            aligned_masks = None
        elif len(item) == 5:
            name, csv_path, dicom_dir, aligned_images, aligned_masks = item
        else:
            raise ValueError("Dataset contract split entries must contain 3 or 5 values.")

        split_name = name.upper()
        print(f"\n{split_name}")
        print(f"  csv: {csv_path}")
        print(f"  dicom_dir: {dicom_dir}")
        print(f"  aligned_images: {aligned_images or 'not configured'}")
        print(f"  aligned_masks: {aligned_masks or 'not configured'}")

        df = read_split(csv_path)
        dicom, aligned_image_map, aligned_mask_map = _source_maps(dicom_dir, aligned_images, aligned_masks)
        image_ids = df["ImageId"].astype(str).unique()

        source_by_id = {}
        missing_ids = []
        for image_id in image_ids:
            source = _choose_source(image_id, dicom, aligned_image_map, aligned_mask_map, source_mode)
            if source is None:
                missing_ids.append(image_id)
            else:
                source_by_id[image_id] = source

        total, positive, negative = split_counts(df)
        actual = {
            "total": total,
            "positive": positive,
            "negative": negative,
            "missing_images": len(missing_ids),
            "aligned_images": sum(1 for s in source_by_id.values() if s == "aligned"),
            "dicom_images": sum(1 for s in source_by_id.values() if s == "dicom"),
        }
        expected = EXPECTED_SPLITS[split_name]
        if not strict_counts:
            expected = dict(actual)

        print(f"  total: {total}")
        print(f"  positive: {positive}")
        print(f"  negative: {negative}")
        print(f"  aligned source: {actual['aligned_images']}")
        print(f"  DICOM source: {actual['dicom_images']}")
        print(f"  missing image source: {actual['missing_images']}")
        if not strict_counts:
            print("  split count contract: CUSTOM/DIAGNOSTIC (strict official counts disabled)")

        mismatches = []
        for key in ("total", "positive", "negative"):
            if actual[key] != expected[key]:
                mismatches.append(f"{key}: expected {expected[key]}, actual {actual[key]}")
        if missing_ids:
            mismatches.append(f"missing image source: expected 0, actual {len(missing_ids)}")

        if mismatches:
            first_missing = f" First missing ImageId: {missing_ids[0]}" if missing_ids else ""
            raise RuntimeError(
                f"{split_name} DATASET CONTRACT FAILED: {'; '.join(mismatches)}.{first_missing}"
            )

        source_values = set(source_by_id.values())
        if source_values == {"aligned"}:
            source_label = "aligned"
        elif source_values == {"dicom"}:
            source_label = "dicom"
        else:
            source_label = "mixed"

        results[split_name] = {
            "actual": actual,
            "expected": expected,
            "csv": Path(csv_path),
            "dicom_dir": Path(dicom_dir),
            "aligned_images": Path(aligned_images) if aligned_images else None,
            "aligned_masks": Path(aligned_masks) if aligned_masks else None,
            "source_mode": source_label,
            "source_by_id": source_by_id,
        }

    train = results["TRAIN"]
    train_df = read_split(train["csv"])
    dicom, aligned_image_map, aligned_mask_map = _source_maps(
        train["dicom_dir"], train["aligned_images"], train["aligned_masks"]
    )
    grouped = train_df.groupby("ImageId")["EncodedPixels"].apply(list)
    positive_ids = [str(i) for i, rles in grouped.items() if any(str(r).strip() not in {"", "-1", "nan"} for r in rles)]
    negative_ids = [str(i) for i, rles in grouped.items() if all(str(r).strip() == "-1" for r in rles)]
    if not positive_ids:
        raise RuntimeError("TRAIN DATASET CONTRACT FAILED: no positive samples are available.")
    if strict_counts and not negative_ids:
        raise RuntimeError("TRAIN DATASET CONTRACT FAILED: both positive and -1 negative samples are required.")

    positive_id = positive_ids[0]
    negative_id = negative_ids[0] if negative_ids else None
    checks = []

    sample_checks = [(positive_id, True)]
    if negative_id is not None:
        sample_checks.append((negative_id, False))
    for image_id, expected_positive in sample_checks:
        source = train["source_by_id"][image_id]
        if source == "aligned":
            _check_aligned_sample(aligned_image_map[image_id], aligned_mask_map[image_id], expected_positive)
            checks.append(f"{image_id}: aligned image/mask readable")
        else:
            dicom_path = dicom[image_id]
            try:
                ds = pydicom.dcmread(dicom_path)
                pixels = ds.pixel_array.astype(np.float32)
                rles = grouped[image_id]
                mask = build_mask_from_rles(rles, pixels.shape)
                if expected_positive and mask.sum() <= 0:
                    raise RuntimeError("Positive RLE decoded to an empty mask.")
                if not expected_positive and mask.sum() != 0:
                    raise RuntimeError("-1 negative sample did not produce an empty mask.")
            except Exception as exc:
                raise RuntimeError(
                    f"TRAIN DATASET CONTRACT FAILED: DICOM/RLE check failed for {image_id}: {exc}"
                ) from exc
            checks.append(f"{image_id}: DICOM/RLE readable")

    print("\nSample source checks")
    for check in checks:
        print(f"  {check}")
    print("\nREADY FOR TRAINING")
    return results
