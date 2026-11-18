"""Reproducibility and forensic evidence for one training run.

This module is observational: it records the existing training/evaluation
behavior without changing the research configuration or model architecture.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

VERSION = 1

REQUIRED_TEST_METRICS = [
    "Test Dice (All)", "Test Dice (Positive)", "Test IoU (All)", "Test IoU (Positive)",
    "Test HD95", "Test Precision", "Test Recall", "Test Specificity", "Test MCC",
    "Test Pixel Accuracy", "Test FPR", "Test FNR", "Image Accuracy", "Image Precision",
    "Image Recall", "Image Specificity", "Image F1", "Image ROC-AUC", "Image PR-AUC",
]


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(getattr(value, "dtype", "")).encode("utf-8"))
        digest.update(str(tuple(getattr(value, "shape", ()))).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def model_signature(model: torch.nn.Module, initial_state_hash: str | None = None) -> dict:
    state = model.state_dict()
    shapes = {key: list(value.shape) for key, value in sorted(state.items()) if hasattr(value, "shape")}
    dtypes = {key: str(value.dtype) for key, value in sorted(state.items()) if hasattr(value, "dtype")}
    signature = {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_shapes": shapes,
        "state_dtypes": dtypes,
        "initial_state_hash": initial_state_hash or state_dict_sha256(state),
    }
    arch_payload = {key: value for key, value in signature.items() if key != "initial_state_hash"}
    signature["architecture_fingerprint"] = hashlib.sha256(
        json.dumps(arch_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return signature


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_rng_state(path: Path, state: dict, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": VERSION, "label": label, "state": state}, path)


def environment_snapshot(device=None) -> dict:
    def version(name):
        try:
            return str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except Exception:
            return None

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "torch": version("torch"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "pydicom": version("pydicom"),
        "matplotlib": version("matplotlib"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "device": str(device) if device is not None else None,
    }
    if torch.cuda.is_available():
        result["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
            }
            for i in range(torch.cuda.device_count())
        ]
    else:
        result["gpus"] = []
    return result


def source_fingerprints(project_root: Path, model_name: str) -> dict:
    names = [
        "train.py", "eval.py", "data_loader.py", "experiment_manager.py", "config.py",
        "visualization.py", "diagnostics.py", "pipeline.py", "research_trace.py",
        "core/plugin_loader.py", f"models/{model_name}.py", "plugins/datasets/siim_acr.py",
    ]
    result = {}
    for name in names:
        path = project_root / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def config_snapshot(args, yaml_data: dict, config_module) -> dict:
    args_data = {k: _json_default(v) for k, v in vars(args).items()}
    keys = [
        "IMAGE_SIZE", "INPUT_CHANNELS", "OUTPUT_CHANNELS", "NORMALIZATION", "BATCH_SIZE",
        "NUM_WORKERS", "EPOCHS", "LEARNING_RATE", "WEIGHT_DECAY", "OPTIMIZER", "SCHEDULER",
        "GRADIENT_ACCUMULATION_STEPS", "SEGMENTATION_THRESHOLD", "IMAGE_LEVEL_MIN_MASK_FRACTION",
        "BCE_WEIGHT", "RANDOM_SEED", "EARLY_STOPPING_ENABLED", "PATIENCE", "MODEL",
    ]
    return {
        "args": args_data,
        "config_module": {key: _json_default(getattr(config_module, key)) for key in keys if hasattr(config_module, key)},
        "yaml": yaml_data,
    }


def dataset_snapshot(args, summary: dict) -> dict:
    root = Path(getattr(args, "data_root", ""))
    files = {}
    for split, attr in (("train", "train_csv_rel"), ("validation", "val_csv_rel"), ("test", "test_csv_rel")):
        rel = getattr(args, attr, None)
        if not rel:
            continue
        path = (root / str(rel)).resolve()
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = sha256_file(path)
            ids = []
            row_count = 0
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row_count += 1
                    if row.get("ImageId"):
                        ids.append(str(row["ImageId"]))
            item["row_count"] = row_count
            item["unique_image_count"] = len(set(ids))
            item["image_id_order_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            item["image_id_set_sha256"] = hashlib.sha256(json.dumps(sorted(set(ids)), separators=(",", ":")).encode()).hexdigest()
        files[split] = item
    return {
        "root": str(root),
        "source_mode": getattr(args, "source_mode", "auto"),
        "aligned_images": str(getattr(args, "aligned_images_dir", "")),
        "aligned_masks": str(getattr(args, "aligned_masks_dir", "")),
        "summary": summary,
        "csv_files": files,
    }


def save_initial_trace(*, run_root: Path, args, yaml_data, config_module, model, device, summary, criterion, optimizer, scheduler, project_root, rng_state):
    audit = run_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    initial_hash = state_dict_sha256(model.state_dict())
    signature = model_signature(model, initial_hash)
    metadata = {
        "version": VERSION,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(map(str, sys.argv)),
        "model": args.model,
        "model_signature": signature,
        "configuration": config_snapshot(args, yaml_data, config_module),
        "dataset": dataset_snapshot(args, summary),
        "criterion": {"class": criterion.__class__.__name__, "repr": repr(criterion)},
        "optimizer": {
            "class": optimizer.__class__.__name__,
            "param_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
        },
        "scheduler": {
            "class": scheduler.__class__.__name__ if scheduler is not None else None,
            "state": scheduler.state_dict() if scheduler is not None else None,
        },
        "environment": environment_snapshot(device),
        "source_fingerprints": source_fingerprints(project_root, args.model),
    }
    write_json(audit / "run_forensics.json", metadata)
    write_json(audit / "model_signature.json", signature)
    write_json(audit / "environment.json", metadata["environment"])
    write_json(audit / "dataset_signature.json", metadata["dataset"])
    (audit / "architecture.txt").write_text(
        "\n".join([
            f"Requested model: {args.model}",
            f"Runtime class: {signature['class']}",
            f"Runtime module: {signature['module']}",
            f"Parameter count: {signature['parameter_count']}",
            f"Trainable parameter count: {signature['trainable_parameter_count']}",
            f"Architecture fingerprint: {signature['architecture_fingerprint']}",
            f"Initial state hash: {signature['initial_state_hash']}",
        ]) + "\n",
        encoding="utf-8",
    )
    (audit / "run_command.txt").write_text(" ".join(map(str, sys.argv)) + "\n", encoding="utf-8")
    save_rng_state(audit / "rng_initial.pt", rng_state, "initial")
    return metadata


def save_rich_checkpoint(path: Path, *, model, metadata, epoch, optimizer, scheduler, training_state, rng_state):
    state = model.state_dict()
    payload = {
        "checkpoint_format": "medplugbench-rich-v1",
        "checkpoint_kind": "best",
        "model_state": state,
        "model_state_sha256": state_dict_sha256(state),
        "model_name": metadata.get("model"),
        "model_class": model.__class__.__name__,
        "model_signature": metadata.get("model_signature"),
        "forensics": metadata,
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "training_state": training_state or {},
        "rng_state": rng_state or capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha256_file(path)


def extract_model_state(loaded):
    if isinstance(loaded, dict) and "model_state" in loaded:
        return loaded["model_state"]
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    return loaded


def save_epoch_trace(run_root: Path, record: dict):
    path = run_root / "audit" / "epoch_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def write_records_csv(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def make_metric_ledger(test_results: dict) -> list[dict]:
    lookup = {
        "Test Dice (All)": test_results.get("all", {}).get("dice"),
        "Test Dice (Positive)": test_results.get("positive", {}).get("dice"),
        "Test IoU (All)": test_results.get("all", {}).get("iou"),
        "Test IoU (Positive)": test_results.get("positive", {}).get("iou"),
        "Test HD95": test_results.get("positive", {}).get("hd95"),
        "Test Precision": test_results.get("all", {}).get("precision"),
        "Test Recall": test_results.get("all", {}).get("recall"),
        "Test Specificity": test_results.get("all", {}).get("specificity"),
        "Test MCC": test_results.get("all", {}).get("mcc"),
        "Test Pixel Accuracy": test_results.get("all", {}).get("pixel_accuracy"),
        "Test FPR": test_results.get("all", {}).get("fpr"),
        "Test FNR": test_results.get("all", {}).get("fnr"),
        "Image Accuracy": test_results.get("image_classification", {}).get("accuracy"),
        "Image Precision": test_results.get("image_classification", {}).get("precision"),
        "Image Recall": test_results.get("image_classification", {}).get("recall"),
        "Image Specificity": test_results.get("image_classification", {}).get("specificity"),
        "Image F1": test_results.get("image_classification", {}).get("f1"),
        "Image ROC-AUC": test_results.get("image_classification", {}).get("roc_auc"),
        "Image PR-AUC": test_results.get("image_classification", {}).get("pr_auc"),
    }
    rows = []
    for metric in REQUIRED_TEST_METRICS:
        value = lookup.get(metric)
        finite = False
        try:
            finite = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            pass
        rows.append({
            "Metric": metric,
            "Status": "OK" if finite else "NOT_AVAILABLE",
            "Value": "" if not finite else float(value),
            "Reason": "Persisted from final test evaluation." if finite else "Metric was missing or non-finite.",
        })
    return rows


def _read_reported_metrics(metrics_path: Path) -> dict[str, float]:
    result = {}
    if not metrics_path.is_file():
        return result
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("Metric", "")).strip()
            if not name:
                continue
            try:
                value = float(row.get("Value", ""))
                if math.isfinite(value):
                    result[name] = value
            except (TypeError, ValueError):
                pass
    return result


def _roc_auc_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _pr_auc_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else float(precision[0])


def recompute_from_records(records: list[dict]) -> dict:
    if not records:
        return {"available": False, "reason": "No per-image prediction records."}

    def f(row, key):
        try:
            return float(row.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    def i(row, key):
        try:
            return int(float(row.get(key, 0)))
        except (TypeError, ValueError):
            return 0

    positive = [r for r in records if i(r, "GroundTruthPositive") == 1]
    all_dice = [f(r, "Dice") for r in records]
    all_iou = [f(r, "IoU") for r in records]
    tp = sum(i(r, "TP") for r in records)
    tn = sum(i(r, "TN") for r in records)
    fp = sum(i(r, "FP") for r in records)
    fn = sum(i(r, "FN") for r in records)
    hd = [f(r, "HD95") for r in positive if math.isfinite(f(r, "HD95"))]
    img_tp = sum(1 for r in records if i(r, "GroundTruthPositive") and i(r, "PredictionPositive"))
    img_tn = sum(1 for r in records if not i(r, "GroundTruthPositive") and not i(r, "PredictionPositive"))
    img_fp = sum(1 for r in records if not i(r, "GroundTruthPositive") and i(r, "PredictionPositive"))
    img_fn = sum(1 for r in records if i(r, "GroundTruthPositive") and not i(r, "PredictionPositive"))
    img_precision = img_tp / (img_tp + img_fp) if img_tp + img_fp else 0.0
    img_recall = img_tp / (img_tp + img_fn) if img_tp + img_fn else 0.0
    img_specificity = img_tn / (img_tn + img_fp) if img_tn + img_fp else 0.0
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if img_precision + img_recall else 0.0
    image_labels = [i(r, "GroundTruthPositive") for r in records]
    image_scores = [f(r, "ImageScore") for r in records]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "available": True,
        "Test Dice (All)": float(np.mean(all_dice)),
        "Test IoU (All)": float(np.mean(all_iou)),
        "Test Dice (Positive)": float(np.mean([f(r, "Dice") for r in positive])) if positive else None,
        "Test IoU (Positive)": float(np.mean([f(r, "IoU") for r in positive])) if positive else None,
        "Test HD95": float(np.mean(hd)) if hd else None,
        "Test Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Test Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Test Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Test MCC": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "Test Pixel Accuracy": (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0,
        "Test FPR": fp / (fp + tn) if fp + tn else 0.0,
        "Test FNR": fn / (fn + tp) if fn + tp else 0.0,
        "Image Accuracy": (img_tp + img_tn) / len(records),
        "Image Precision": img_precision,
        "Image Recall": img_recall,
        "Image Specificity": img_specificity,
        "Image F1": img_f1,
        "Image ROC-AUC": _roc_auc_from_scores(image_labels, image_scores),
        "Image PR-AUC": _pr_auc_from_scores(image_labels, image_scores),
        "prediction_count": len(records),
        "empty_prediction_rate": sum(1 for r in records if i(r, "PredPositivePixels") == 0) / len(records),
        "mean_predicted_fraction": float(np.mean([f(r, "PredPositiveFraction") for r in records])),
        "mean_target_fraction": float(np.mean([f(r, "GTPositiveFraction") for r in records])),
    }


def prediction_fingerprint(records: list[dict]) -> str | None:
    if not records:
        return None
    keep = []
    keys = [
        "SampleIndex", "ImageId", "GroundTruthPositive", "PredictionPositive", "GTPositivePixels", "PredPositivePixels",
        "MeanProbability", "MaxProbability", "MinProbability", "LogitMean", "LogitMax", "LogitMin",
        "PredictionMaskHash", "ProbabilityMapHash", "Dice", "IoU", "TP", "TN", "FP", "FN", "ImageScore",
    ]
    for row in records:
        keep.append({key: row.get(key) for key in keys})
    payload = json.dumps(keep, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_metrics(metrics_path: Path, records: list[dict], tolerance: float = 1e-5) -> dict:
    recomputed = recompute_from_records(records)
    if not recomputed.get("available"):
        return {"status": "FAILED", "reason": recomputed.get("reason"), "prediction_fingerprint": None}
    reported = _read_reported_metrics(metrics_path)
    mismatches = []
    for key in REQUIRED_TEST_METRICS:
        expected = recomputed.get(key)
        actual = reported.get(key)
        if expected is None or actual is None:
            continue
        if abs(float(actual) - float(expected)) > tolerance:
            mismatches.append({"metric": key, "reported": actual, "recomputed": expected})
    return {
        "status": "PASS" if not mismatches else "FAILED",
        "reported_metrics_count": len(reported),
        "prediction_fingerprint": prediction_fingerprint(records),
        "recomputed": recomputed,
        "mismatches": mismatches,
    }


def duplicate_signals(experiments_root: Path, current_run_id: str) -> dict:
    histories = {}
    predictions = {}
    for run_root in experiments_root.glob("RUN_*"):
        if run_root.name == current_run_id:
            continue
        history = run_root / "logs" / "training_log.csv"
        cases = run_root / "audit" / "test_predictions.csv"
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        item = {"run_id": run_root.name, "model": manifest.get("model", "unknown")}
        if history.is_file():
            histories.setdefault(sha256_file(history), []).append(item)
        if cases.is_file():
            predictions.setdefault(sha256_file(cases), []).append(item)
    return {
        "duplicate_training_histories": [v for v in histories.values() if len(v) > 1],
        "duplicate_prediction_records": [v for v in predictions.values() if len(v) > 1],
    }


def generate_audit_graphs(run_root: Path, history_rows: list[dict], train_epoch_records: list[dict], test_records: list[dict]):
    audit = run_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    epochs = []
    train_loss = []
    val_dice = []
    val_pos_dice = []
    lrs = []
    probs = []
    logits = []
    grad = []
    pred_frac = []
    target_frac = []
    for row in history_rows:
        try: epochs.append(int(row.get("Epoch", "")))
        except (ValueError, TypeError): continue
        def num(key):
            try: return float(row.get(key, "nan"))
            except (TypeError, ValueError): return float("nan")
        train_loss.append(num("TrainLoss")); val_dice.append(num("ValidationDiceAll")); val_pos_dice.append(num("ValidationDicePositive")); lrs.append(num("LearningRate"))
    for row in train_epoch_records:
        stats = row.get("train", {})
        probs.append(float(stats.get("probability_mean", 0.0)))
        logits.append(float(stats.get("output_mean", 0.0)))
        grad.append(float(stats.get("grad_norm_mean", 0.0)))
        pred_frac.append(float(stats.get("prediction_positive_fraction", 0.0)))
        target_frac.append(float(stats.get("target_positive_fraction", 0.0)))

    if epochs:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs, train_loss, label="Train loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "training_loss_curve_audit.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs, val_dice, label="Validation Dice (all)"); ax.plot(epochs, val_pos_dice, label="Validation Dice (positive)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Dice"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "validation_dice_curve_audit.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs, lrs, label="Learning rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "learning_rate_curve.png", dpi=150); plt.close(fig)
        if probs:
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs[:len(probs)], probs, label="Mean probability"); ax.set_xlabel("Epoch"); ax.set_ylabel("Probability"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "probability_stats.png", dpi=150); plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs[:len(logits)], logits, label="Mean logit"); ax.set_xlabel("Epoch"); ax.set_ylabel("Logit"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "logit_stats.png", dpi=150); plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs[:len(grad)], grad, label="Mean gradient norm"); ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient norm"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "gradient_stability.png", dpi=150); plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(epochs[:len(pred_frac)], pred_frac, label="Predicted positive fraction"); ax.plot(epochs[:len(target_frac)], target_frac, label="Target positive fraction"); ax.set_xlabel("Epoch"); ax.set_ylabel("Pixel fraction"); ax.legend(); fig.tight_layout(); fig.savefig(audit / "prediction_behavior.png", dpi=150); plt.close(fig)

    if test_records:
        tp = sum(int(float(r.get("TP", 0))) for r in test_records)
        tn = sum(int(float(r.get("TN", 0))) for r in test_records)
        fp = sum(int(float(r.get("FP", 0))) for r in test_records)
        fn = sum(int(float(r.get("FN", 0))) for r in test_records)
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
        fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(matrix); ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["True 0", "True 1"]); ax.set_title("Pixel confusion matrix");
        for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, f"{value:.0f}", ha="center", va="center")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(audit / "test_confusion_matrix.png", dpi=150); plt.close(fig)


def write_audit_bundle(*, run_root: Path, metadata: dict, test_results: dict, validation_records: list[dict], test_records: list[dict], history_rows: list[dict], epoch_records: list[dict], metrics_path: Path, best_checkpoint_path: Path, resume_checkpoint_path: Path, experiments_root: Path):
    audit = run_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    write_json(audit / "metric_status.json", {row["Metric"]: row["Status"] for row in make_metric_ledger(test_results)})
    with (audit / "metric_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Status", "Value", "Reason"])
        writer.writeheader(); writer.writerows(make_metric_ledger(test_results))
    write_records_csv(audit / "test_predictions.csv", test_records)
    write_records_csv(audit / "validation_predictions.csv", validation_records)
    write_json(audit / "training_history.json", history_rows)
    write_json(audit / "dataset_stats.json", metadata.get("dataset", {}))
    verification = verify_metrics(metrics_path, test_records)
    verification["cross_run_duplicates"] = duplicate_signals(experiments_root, run_root.name)
    write_json(audit / "verification.json", verification)

    hashes = {"run_id": run_root.name}
    for label, path in (("best", best_checkpoint_path), ("resume", resume_checkpoint_path)):
        if path.is_file():
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            state = extract_model_state(loaded)
            hashes[label] = {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "state_sha256": state_dict_sha256(state) if isinstance(state, dict) else None,
                "model_name": loaded.get("model_name") if isinstance(loaded, dict) else None,
                "model_class": loaded.get("model_class") if isinstance(loaded, dict) else None,
                "epoch": loaded.get("epoch") if isinstance(loaded, dict) else None,
            }
    write_json(audit / "checkpoint_hashes.json", hashes)
    if best_checkpoint_path.is_file():
        loaded = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        write_json(audit / "best_checkpoint_metadata.json", {
            "path": str(best_checkpoint_path.resolve()),
            "sha256": sha256_file(best_checkpoint_path),
            "format": loaded.get("checkpoint_format", "legacy") if isinstance(loaded, dict) else "legacy",
            "model_name": loaded.get("model_name") if isinstance(loaded, dict) else None,
            "model_class": loaded.get("model_class") if isinstance(loaded, dict) else None,
            "epoch": loaded.get("epoch") if isinstance(loaded, dict) else None,
            "embedded_forensics": bool(isinstance(loaded, dict) and loaded.get("forensics")),
        })
    generate_audit_graphs(run_root, history_rows, epoch_records, test_records)
    guide = (
        f"# Reproducibility Guide\n\nRun: `{run_root.name}`\nModel: `{metadata.get('model')}`\n\n"
        "The run stores configuration, environment, dataset fingerprints, runtime model identity, "
        "RNG state, checkpoints, epoch diagnostics, per-image predictions, metric completeness, "
        "independent metric verification and forensic graphs.\n"
    )
    (audit / "REPRODUCIBILITY_GUIDE.md").write_text(guide, encoding="utf-8")
    write_json(audit / "artifact_verification.json", {
        "required": [
            "config", "metrics", "training_log", "checkpoint", "resume_checkpoint", "report",
            "metric_ledger", "test_predictions", "validation_predictions", "verification",
            "checkpoint_hashes", "checkpoint_metadata", "model_signature", "dataset_signature",
            "training_history", "rng_initial", "rng_final", "architecture", "run_command",
            "dataset_stats", "prediction_behavior.png", "learning_rate_curve.png",
            "probability_stats.png", "logit_stats.png", "gradient_stability.png",
            "test_confusion_matrix.png", "training_loss_curve_audit.png",
            "validation_dice_curve_audit.png", "qualitative_cache.npz", "REPRODUCIBILITY_GUIDE.md",
        ],
        "present": sorted(str(p.relative_to(run_root)) for p in run_root.rglob("*") if p.is_file()),
    })
    return verification
