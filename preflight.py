"""Cheap pre-training checks. This script never starts model training."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

import config
from core.plugin_loader import load_config_yaml


def _check_training_environment(device):
    """Import the actual training entry point without starting training."""
    command = [sys.executable, "train.py", "--help"]
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Training environment check timed out while importing train.py.") from exc
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            "Training environment check failed. train.py could not start in help mode.\n"
            + output.strip()
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device == "auto":
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    elif device == "cuda":
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  PyTorch: {torch.__version__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    from dataset_contract import validate_contract

    yaml_data = load_config_yaml(args.config)
    dataset_cfg = yaml_data["dataset"]
    training_cfg = yaml_data.get("training", {})
    device = args.device or training_cfg.get("device", config.DEVICE)
    root_value = os.path.expandvars(os.path.expanduser(str(dataset_cfg["root"])))
    if not root_value or root_value.startswith("${"):
        raise FileNotFoundError(
            "dataset.root is not configured. Set MEDPLUG_DATASET_ROOT in PowerShell or use the UI Dataset root field."
        )
    root = Path(root_value).resolve()
    print(f"  dataset root: {root}")
    aligned_images = root / dataset_cfg.get("aligned_images", "aligned_dataset/images")
    aligned_masks = root / dataset_cfg.get("aligned_masks", "aligned_dataset/masks")
    source_mode = str(dataset_cfg.get("source_mode", "auto"))
    strict_counts = bool(dataset_cfg.get("strict_counts", True))
    splits = [
        ("TRAIN", root / dataset_cfg.get("train_csv", "data/train-rle.csv"), root / dataset_cfg.get("train_images", "images/dicom-images-train"), aligned_images, aligned_masks),
        ("VALIDATION", root / dataset_cfg.get("val_csv", "data/val-rle.csv"), root / dataset_cfg.get("val_images", dataset_cfg.get("train_images", "images/dicom-images-train")), aligned_images, aligned_masks),
        ("TEST", root / dataset_cfg.get("test_csv", "data/test-rle.csv"), root / dataset_cfg.get("test_images", "images/dicom-images-test"), aligned_images, aligned_masks),
    ]

    print("=== Pre-Training Dataset Contract Check ===")
    print(f"  strict official split counts: {strict_counts}")
    results = validate_contract(splits, source_mode=source_mode, strict_counts=strict_counts)

    print("Training Environment Check")
    _check_training_environment(device)

    print("\nREADY FOR TRAINING")
    return results


if __name__ == "__main__":
    main()
