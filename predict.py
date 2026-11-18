"""Run a trained segmentation model on an image and generate a professional PDF report.

Examples:
    python predict.py --image path/to/xray.dcm --model unet --checkpoint path/to/unet_best_model.pth
    python predict.py --image path/to/xray.png --run-id RUN_20260904_083000_A1B2C
    python predict.py --image path/to/xray.dcm --all-models
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from core.plugin_loader import load_model_plugin
from experiment_manager import ExperimentManager
from prediction_report import analyze_image_quality, generate_prediction_report, generate_combined_prediction_report
from research_trace import extract_model_state

try:
    from PIL import Image
except ImportError:  # Pillow is already commonly installed with matplotlib, but make the message clear.
    Image = None

MODELS = ("unet", "attention_unet", "transunet", "segformer")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pneumothorax segmentation and generate a PDF report.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a DICOM, PNG, JPG, JPEG, or BMP image. Omit it to use the drag-and-drop desktop picker.")
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None, help="Use the checkpoint and configuration from a completed run.")
    parser.add_argument("--all-models", action="store_true", help="Run every model with an available completed checkpoint and make one comparison report.")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional binary ground-truth mask for per-image Dice/IoU/precision/recall/specificity/MCC/accuracy/FPR/FNR/HD95.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=config.SEGMENTATION_THRESHOLD)
    parser.add_argument("--image-size", nargs=2, type=int, default=list(config.IMAGE_SIZE), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--input-channels", choices=(1, 3), type=int, default=None)
    parser.add_argument("--normalization", choices=("minmax", "imagenet"), default=None)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_image(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        import pydicom

        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)
    else:
        if Image is None:
            raise RuntimeError("Pillow is required for PNG/JPG/BMP prediction. Install pillow.")
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}.")
    return image


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - low) / (high - low)).astype(np.float32)


def resize_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a model-resolution binary mask back to the source image size."""
    import torch.nn.functional as F

    mask = np.asarray(mask, dtype=bool)
    if tuple(mask.shape) == tuple(image_shape):
        return mask
    return (
        F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=image_shape,
            mode="nearest",
        )[0, 0]
        .numpy()
        >= 0.5
    )


def preprocess(image: np.ndarray, image_size: tuple[int, int], input_channels: int, normalization: str):
    import torch.nn.functional as F

    norm = normalize_minmax(image)
    tensor = torch.from_numpy(norm).float()[None, None, ...]
    if tuple(image.shape) != tuple(image_size):
        tensor = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    if input_channels == 3:
        tensor = tensor.repeat(3, 1, 1)
        if normalization == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
            tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def load_manifest_config(run_id: str):
    manager = ExperimentManager()
    paths = manager.get_paths(run_id)
    manifest = manager.load_manifest(paths)
    snapshot = manifest.get("configuration") or {}
    return manager, paths, manifest, snapshot


def _checkpoint_candidates(manager: ExperimentManager, run: dict) -> list[Path]:
    """Find a trained checkpoint even when an older manifest used an absolute path."""
    model = str(run.get("model", "")).strip()
    run_id = str(run.get("run_id", "")).strip()
    checkpoint_name = str(run.get("checkpoint_name") or f"{model}_best_model")
    if not checkpoint_name.lower().endswith(".pth"):
        checkpoint_name += ".pth"

    candidates: list[Path] = []
    try:
        paths = manager.get_paths(run_id)
        candidates.append(paths.checkpoint_path)
    except Exception:
        paths = None

    artifacts = run.get("artifacts") or {}
    artifact_value = artifacts.get("checkpoint")
    if artifact_value:
        artifact_path = Path(str(artifact_value))
        if paths is not None and not artifact_path.is_absolute():
            candidates.append(paths.root / artifact_path)
        candidates.append(artifact_path)

    cfg = run.get("configuration") or {}
    configured_dir = cfg.get("checkpoint_dir") or cfg.get("checkpoint_directory")
    if configured_dir:
        candidates.append(Path(str(configured_dir)) / checkpoint_name)

    # Current project-level location. This also works when runs were copied
    # to a new machine but the old absolute checkpoint_dir in the manifest
    # no longer exists.
    candidates.append(config.CHECKPOINT_DIRECTORY / checkpoint_name)
    candidates.append(Path.cwd() / "checkpoints" / checkpoint_name)

    unique = []
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.expanduser()
        except (OSError, RuntimeError):
            continue
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_checkpoint_for_run(manager: ExperimentManager, run: dict) -> Path | None:
    for candidate in _checkpoint_candidates(manager, run):
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_latest_runs() -> dict[str, tuple[str, Path, dict]]:
    manager = ExperimentManager()
    found = {}
    for run in manager.list_runs():
        if str(run.get("status", "")).upper() not in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}:
            continue
        model = str(run.get("model", ""))
        if model not in MODELS:
            continue
        run_id = str(run.get("run_id", ""))
        checkpoint = find_checkpoint_for_run(manager, run)
        if checkpoint is None:
            continue
        previous = found.get(model)
        if previous is None or run_id > previous[0]:
            found[model] = (run_id, checkpoint, run)
    return found


def checkpoint_for_request(args, model_name: str):
    if args.run_id:
        manager, paths, manifest, snapshot = load_manifest_config(args.run_id)
        if model_name != str(manifest.get("model", model_name)):
            raise ValueError(f"Run {args.run_id} belongs to model '{manifest.get('model')}', not '{model_name}'.")
        checkpoint = find_checkpoint_for_run(manager, manifest)
        if checkpoint is None:
            expected = ", ".join(str(p) for p in _checkpoint_candidates(manager, manifest)[:4])
            raise FileNotFoundError(
                f"No checkpoint file was found for run {args.run_id}. Checked the run folder and configured checkpoint locations. "
                f"Expected one of: {expected}"
            )
        return checkpoint, snapshot
    if args.checkpoint:
        return Path(args.checkpoint), {}
    latest = discover_latest_runs().get(model_name)
    if latest is None:
        raise FileNotFoundError(f"No completed checkpoint found for model '{model_name}'.")
    return latest[1], latest[2].get("configuration") or {}


def _checkpoint_input_channels(state: dict) -> int | None:
    for key, value in state.items():
        if not hasattr(value, "shape") or len(value.shape) != 4:
            continue
        name = key.removeprefix("module.")
        if name.endswith("enc1.block.0.weight") or name.endswith("enc1.conv1.weight"):
            return int(value.shape[1])
    return None


def load_checkpoint(model_name: str, checkpoint: Path, in_channels: int | None, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state = extract_model_state(state)
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    detected_channels = _checkpoint_input_channels(state) if isinstance(state, dict) else None
    effective_channels = detected_channels or in_channels or 1
    model = load_model_plugin(model_name, in_channels=effective_channels, out_channels=1, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, effective_channels


def make_comparison():
    # The comparison table is sourced from the persisted benchmark CSV, one latest run per model.
    csv_path = config.EXPERIMENTS_DIRECTORY / config.BENCHMARK_RESULTS_FILE
    if not csv_path.is_file():
        return []
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty or "Model" not in df.columns:
        return []
    if "Run ID" in df.columns:
        df = df.sort_values("Run ID").drop_duplicates("Model", keep="last")
    metric_map = {
        "Validation Dice": "validation_dice",
        "Test Dice": "dice",
        "Test IoU": "iou",
        "Test HD95": "hd95",
        "Test Precision": "precision",
        "Test Recall": "recall",
        "Test Specificity": "specificity",
        "Test MCC": "mcc",
        "Image F1": "image_f1",
        "Image ROC-AUC": "image_roc_auc",
        "Epoch": "epoch",
        "Training Time": "training_time",
    }
    result = []
    for _, row in df.iterrows():
        item = {"model": str(row.get("Model", "unknown"))}
        for source, target in metric_map.items():
            try:
                item[target] = float(row.get(source))
            except (TypeError, ValueError):
                item[target] = float("nan")
        result.append(item)
    return result


def load_ground_truth(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")
    if Image is None:
        raise RuntimeError("Pillow is required for ground-truth masks.")
    gt = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    gt = resize_mask_to_image(gt >= 128, image_shape)
    return gt


def single_image_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    eps = 1e-6
    dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
    iou = float((tp + eps) / (tp + fp + fn + eps))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0
    from eval import hd95_single
    hd95 = float(hd95_single(pred, gt)) if gt.any() else float("nan")
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "specificity": specificity, "mcc": mcc, "pixel_accuracy": accuracy, "fpr": fpr, "fnr": fnr, "hd95": hd95, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def predict_one(
    *,
    image: np.ndarray,
    image_name: str,
    model_name: str,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float,
    image_size: tuple[int, int],
    input_channels: int,
    normalization: str,
    comparison: list[dict],
    ground_truth: np.ndarray | None = None,
):
    quality = analyze_image_quality(image)
    # Build the model first. Older runs may have been trained with 1 channel
    # while the current config defaults to 3 channels.
    model, detected_channels = load_checkpoint(model_name, checkpoint, input_channels, device)
    input_channels = detected_channels
    batch = preprocess(image, image_size, input_channels, normalization).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
        probability = torch.sigmoid(logits)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - start) * 1000.0

    probability_map = probability[0, 0].detach().cpu().numpy()
    model_mask = probability_map >= threshold

    # The model predicts at the configured inference size (usually 256x256).
    # Reports and saved overlays must match the original X-ray dimensions,
    # otherwise a 256x256 mask can cover only part of a 1024x1024 image.
    mask = resize_mask_to_image(model_mask, tuple(image.shape))

    image_probability = float(probability_map.mean())
    peak_probability = float(probability_map.max())
    per_image = single_image_metrics(mask, ground_truth) if ground_truth is not None else None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    safe_model = model_name.replace("/", "_")
    pdf_path = output_dir / f"{stem}_{safe_model}_report.pdf"
    mask_path = output_dir / f"{stem}_{safe_model}_mask.png"
    overlay_path = output_dir / f"{stem}_{safe_model}_overlay.png"
    json_path = output_dir / f"{stem}_{safe_model}_result.json"

    # Save prediction images with the exact same visual semantics used in the PDF.
    import matplotlib.pyplot as plt
    from prediction_report import _display_image, boundary

    display = _display_image(image)
    plt.imsave(mask_path, mask.astype(float), cmap="gray", vmin=0, vmax=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display, cmap="gray")
    ax.imshow(mask, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    edges = boundary(mask)
    if edges.any():
        ax.contour(edges, levels=[0.5], colors="lime", linewidths=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(overlay_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    generate_prediction_report(
        output_pdf=pdf_path,
        image=image,
        quality=quality,
        model_name=model_name,
        image_name=image_name,
        prediction_mask=mask,
        probability_map=probability_map,
        image_probability=image_probability,
        peak_probability=peak_probability,
        inference_ms=inference_ms,
        threshold=threshold,
        image_level_fraction=config.IMAGE_LEVEL_MIN_MASK_FRACTION,
        comparison=comparison,
        per_image_metrics=per_image,
    )

    payload = {
        "image": str(image_name),
        "model": model_name,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "prediction": "Positive" if float(mask.mean()) >= config.IMAGE_LEVEL_MIN_MASK_FRACTION else "Negative",
        "mean_pixel_probability": image_probability,
        "peak_pixel_probability": peak_probability,
        "threshold": threshold,
        "predicted_area_fraction": float(mask.mean()),
        "predicted_area_percent": float(mask.mean() * 100.0),
        "inference_time_ms": inference_ms,
        "quality": quality.__dict__,
        "per_image_metrics": per_image,
        "report": str(pdf_path),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "json": str(json_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def launch_drag_drop_ui():
    """Launch a small desktop drag-and-drop inference window."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for the desktop predictor.") from exc

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        drag_drop_enabled = True
    except ImportError:
        # Keep the predictor usable even when the optional DnD package is missing.
        root = tk.Tk()
        drag_drop_enabled = False
    root.title("Vessel Vision — X-ray Prediction")
    root.geometry("760x560")
    root.minsize(680, 500)
    root.configure(bg="#f3f6fa")

    selected_path = {"value": None}
    status = tk.StringVar(value="Drop an X-ray image here")
    model_var = tk.StringVar(value="unet")
    all_models_var = tk.BooleanVar(value=True)
    threshold_var = tk.StringVar(value=f"{config.SEGMENTATION_THRESHOLD:.2f}")

    header = tk.Frame(root, bg="#0f172a", height=76)
    header.pack(fill="x")
    tk.Label(
        header,
        text="Vessel Vision",
        bg="#0f172a",
        fg="white",
        font=("Segoe UI", 20, "bold"),
    ).pack(anchor="w", padx=28, pady=(16, 0))
    tk.Label(
        header,
        text="X-ray segmentation • research prototype",
        bg="#0f172a",
        fg="#cbd5e1",
        font=("Segoe UI", 10),
    ).pack(anchor="w", padx=29, pady=(2, 12))

    body = tk.Frame(root, bg="#f3f6fa")
    body.pack(fill="both", expand=True, padx=26, pady=24)

    drop = tk.Label(
        body,
        text=(
            "DRAG & DROP X-RAY HERE\n\nDICOM • PNG • JPG • JPEG • BMP"
            if drag_drop_enabled
            else "CHOOSE X-RAY FILE\n\nDICOM • PNG • JPG • JPEG • BMP"
        ),
        relief="ridge",
        bd=2,
        bg="white",
        fg="#334155",
        font=("Segoe UI", 15, "bold"),
        height=8,
    )
    drop.pack(fill="x", pady=(0, 8))

    browse_row = tk.Frame(body, bg="#f3f6fa")
    browse_row.pack(fill="x", pady=(0, 14))

    path_label = tk.Label(body, text="No image selected", bg="#f3f6fa", fg="#64748b", font=("Segoe UI", 9))
    path_label.pack(fill="x", pady=(0, 18))

    options = tk.Frame(body, bg="#f3f6fa")
    options.pack(fill="x")

    tk.Label(options, text="Model", bg="#f3f6fa", fg="#334155", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
    model_menu = tk.OptionMenu(options, model_var, *MODELS)
    model_menu.config(bg="white", relief="solid", bd=1, font=("Segoe UI", 10), width=18)
    model_menu.grid(row=1, column=0, sticky="ew", padx=(0, 18))

    tk.Label(options, text="Threshold", bg="#f3f6fa", fg="#334155", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=6)
    tk.Entry(options, textvariable=threshold_var, font=("Segoe UI", 10), relief="solid", bd=1).grid(row=1, column=1, sticky="ew", padx=(0, 18))

    tk.Checkbutton(
        options,
        text="Run all available models",
        variable=all_models_var,
        bg="#f3f6fa",
        fg="#334155",
        activebackground="#f3f6fa",
        selectcolor="white",
        font=("Segoe UI", 10),
    ).grid(row=1, column=2, sticky="w")
    options.columnconfigure(0, weight=2)
    options.columnconfigure(1, weight=1)
    options.columnconfigure(2, weight=2)

    result_box = tk.Text(body, height=8, bg="#0b1220", fg="#e2e8f0", relief="flat", font=("Consolas", 9))
    result_box.pack(fill="both", expand=True, pady=(22, 12))
    result_box.configure(state="disabled")

    def show_result(text):
        result_box.configure(state="normal")
        result_box.delete("1.0", "end")
        result_box.insert("end", text)
        result_box.configure(state="disabled")

    def set_file(path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            return
        selected_path["value"] = path
        path_label.configure(text=str(path), fg="#0f172a")
        status.set(f"Selected: {path.name}")

    def on_drop(event):
        if not drag_drop_enabled:
            return
        try:
            paths = root.tk.splitlist(event.data)
        except tk.TclError:
            paths = [event.data]
        if paths:
            set_file(paths[0])

    def browse_file():
        path = filedialog.askopenfilename(
            title="Select X-ray image",
            filetypes=[
                ("X-ray / image files", "*.dcm *.dicom *.png *.jpg *.jpeg *.bmp"),
                ("DICOM files", "*.dcm *.dicom"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            set_file(path)

    if drag_drop_enabled:
        drop.drop_target_register(DND_FILES)
        drop.dnd_bind("<<Drop>>", on_drop)
        status.set("Drop an X-ray image here, or use Browse")
    else:
        status.set("Drag & drop is unavailable; use Browse to select an X-ray")

    tk.Button(
        browse_row,
        text="Browse X-ray…",
        command=browse_file,
        bg="white",
        fg="#0f172a",
        activebackground="#e2e8f0",
        relief="solid",
        bd=1,
        font=("Segoe UI", 10, "bold"),
        padx=16,
        pady=8,
        cursor="hand2",
    ).pack(side="left")

    tk.Label(
        browse_row,
        text="or drag a file onto the box above",
        bg="#f3f6fa",
        fg="#64748b",
        font=("Segoe UI", 9),
    ).pack(side="left", padx=12)

    def run_prediction():
        image_path = selected_path["value"]
        if image_path is None:
            messagebox.showwarning("No image", "Choose an X-ray with Browse or drag one into the drop area.")
            return
        try:
            threshold = float(threshold_var.get())
            if not 0 <= threshold <= 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Threshold", "Threshold must be between 0 and 1.")
            return

        output_dir = Path(__file__).resolve().parent / "predictions"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--threshold",
            str(threshold),
            "--device",
            "auto",
        ]
        if all_models_var.get():
            command.append("--all-models")
        else:
            command.extend(["--model", model_var.get()])
        try:
            import subprocess
            result = subprocess.run(command, cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=1800)
            text = ((result.stdout or "") + (result.stderr or "")).strip()
            show_result(text or "Prediction completed.")
            status.set("Prediction completed" if result.returncode == 0 else "Prediction failed")
            if result.returncode == 0:
                messagebox.showinfo("Prediction complete", "Reports and visual outputs were saved in the predictions folder.")
            else:
                messagebox.showerror("Prediction failed", text[-1200:] if text else "Unknown error")
        except Exception as exc:
            show_result(f"{type(exc).__name__}: {exc}")
            status.set("Prediction failed")
            messagebox.showerror("Prediction failed", str(exc))

    tk.Label(body, textvariable=status, bg="#f3f6fa", fg="#475569", font=("Segoe UI", 9)).pack(anchor="w")
    tk.Button(
        body,
        text="Generate Prediction & Report",
        command=run_prediction,
        bg="#0f172a",
        fg="white",
        activebackground="#1e293b",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11, "bold"),
        padx=18,
        pady=10,
        cursor="hand2",
    ).pack(anchor="e", pady=(10, 0))

    root.mainloop()


def main():
    args = parse_args()
    if args.image is None:
        launch_drag_drop_ui()
        return
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    device = resolve_device(args.device)
    image = load_image(image_path)
    ground_truth = load_ground_truth(args.ground_truth, tuple(image.shape)) if args.ground_truth else None
    comparison = make_comparison()

    jobs = []
    if args.all_models:
        latest = discover_latest_runs()
        if not latest:
            manager = ExperimentManager()
            completed = [r for r in manager.list_runs() if str(r.get("status", "")).upper() in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}]
            models = sorted({str(r.get("model", "unknown")) for r in completed if r.get("model")})
            details = ", ".join(models) if models else "none"
            raise FileNotFoundError(
                "No trained checkpoint files were found for --all-models. "
                f"Completed runs in this project: {details}. "
                "Make sure the .pth files from training are present in each run's checkpoints folder or in the configured checkpoints directory."
            )
        for model_name in MODELS:
            if model_name not in latest:
                continue
            run_id, checkpoint, run = latest[model_name]
            snapshot = run.get("configuration") or {}
            jobs.append((model_name, checkpoint, snapshot))
    else:
        model_name = args.model
        if not model_name and args.run_id:
            _, _, manifest, _ = load_manifest_config(args.run_id)
            model_name = str(manifest.get("model"))
        if not model_name:
            raise ValueError("Provide --model, --run-id, or use --all-models.")
        checkpoint, snapshot = checkpoint_for_request(args, model_name)
        jobs.append((model_name, checkpoint, snapshot))

    results = []
    for model_name, checkpoint, snapshot in jobs:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            print(f"Skipping {model_name}: checkpoint not found at {checkpoint}")
            continue
        # X-ray checkpoints are normally trained as grayscale. The loader also
        # inspects the checkpoint so old/new 1-channel and 3-channel runs both work.
        in_channels = args.input_channels or int(snapshot.get("input_channels") or 1)
        normalization = args.normalization or str(snapshot.get("normalization") or "minmax")
        result = predict_one(
            image=image,
            image_name=image_path.name,
            model_name=model_name,
            checkpoint=checkpoint,
            output_dir=args.output_dir,
            device=device,
            threshold=float(args.threshold),
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            input_channels=in_channels,
            normalization=normalization,
            comparison=comparison,
            ground_truth=ground_truth,
        )
        results.append(result)
        print(f"[{model_name}] {result['prediction']}  mean_pixel_probability={result['mean_pixel_probability']:.4f}  inference={result['inference_time_ms']:.2f} ms")
        print(f"  Report:  {result['report']}")
        print(f"  Overlay: {result['overlay']}")

    if not results:
        raise RuntimeError("No prediction was generated.")

    if args.all_models:
        summary_path = args.output_dir / f"{image_path.stem}_all_models_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        combined_pdf = args.output_dir / f"{image_path.stem}_all_models_report.pdf"
        generate_combined_prediction_report(
            output_pdf=combined_pdf, image=image, quality=analyze_image_quality(image),
            image_name=image_path.name, results=results, comparison=comparison, threshold=float(args.threshold),
        )
        print(f"Comparison summary: {summary_path}")
        print(f"Combined report: {combined_pdf}")


if __name__ == "__main__":
    main()
