"""Prediction quality analysis and professional PDF reporting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


@dataclass
class ImageQuality:
    status: str
    rows: int
    columns: int
    aspect_ratio: float
    minimum: float
    maximum: float
    mean: float
    std: float
    p01: float
    p99: float
    dynamic_range: float
    black_clipping_fraction: float
    white_clipping_fraction: float
    blur_score_laplacian: float
    edge_density: float
    noise_proxy: float
    contrast_proxy: float
    blur_quality: float
    edge_quality: float
    contrast_quality: float
    noise_quality: float


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[1:-1, 1:-1]
    )
    return mask & ~eroded


def _shift_mean(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    total = np.zeros_like(image, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return total / 9.0


def analyze_image_quality(image: np.ndarray) -> ImageQuality:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Quality analysis expects a single-channel 2D image.")

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    mean = float(np.mean(image))
    std = float(np.std(image))
    p01, p99 = np.percentile(image, [1, 99])
    dynamic_range = float(maximum - minimum)

    # Discrete Laplacian variance: a simple sharpness proxy without OpenCV.
    padded = np.pad(image, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    blur_score = float(np.var(laplacian))

    gx = np.diff(image, axis=1, append=image[:, -1:])
    gy = np.diff(image, axis=0, append=image[-1:, :])
    gradient = np.hypot(gx, gy)
    edge_threshold = float(np.percentile(gradient, 75))
    edge_density = float(np.mean(gradient > edge_threshold)) if edge_threshold > 0 else 0.0

    smooth = _shift_mean(image)
    residual = image - smooth
    noise_proxy = float(np.std(residual)) / max(std, 1e-6)
    contrast_proxy = float(std / max(mean, 1e-6))

    # These are research-oriented visual quality scores, not clinical scores.
    blur_quality = 100.0 * blur_score / (blur_score + 10.0)
    edge_quality = 100.0 * min(edge_density / 0.03, 1.0)
    contrast_quality = 100.0 * min(std / 90.0, 1.0)
    noise_quality = 100.0 * max(0.0, 1.0 - noise_proxy / 2.0)

    black_fraction = float(np.mean(image <= minimum + 1e-6))
    white_fraction = float(np.mean(image >= maximum - 1e-6))
    status = "PASS"
    if blur_quality < 40 or noise_quality < 35 or black_fraction > 0.02 or white_fraction > 0.02:
        status = "REVIEW"

    return ImageQuality(
        status=status,
        rows=int(image.shape[0]),
        columns=int(image.shape[1]),
        aspect_ratio=float(image.shape[1] / max(image.shape[0], 1)),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        p01=float(p01),
        p99=float(p99),
        dynamic_range=dynamic_range,
        black_clipping_fraction=black_fraction,
        white_clipping_fraction=white_fraction,
        blur_score_laplacian=blur_score,
        edge_density=edge_density,
        noise_proxy=noise_proxy,
        contrast_proxy=contrast_proxy,
        blur_quality=blur_quality,
        edge_quality=edge_quality,
        contrast_quality=contrast_quality,
        noise_quality=noise_quality,
    )


def _display_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _quality_rows(quality: ImageQuality):
    return [
        ("Quality Status", quality.status),
        ("Rows", f"{quality.rows}"),
        ("Columns", f"{quality.columns}"),
        ("Aspect Ratio", f"{quality.aspect_ratio:.4f}"),
        ("Blur / Sharpness", f"{quality.blur_quality:.1f}/100"),
        ("Edge Density", f"{quality.edge_quality:.1f}/100"),
        ("Contrast", f"{quality.contrast_quality:.1f}/100"),
        ("Noise Quality", f"{quality.noise_quality:.1f}/100"),
        ("Black Clipping", f"{(1 - quality.black_clipping_fraction) * 100:.1f}/100"),
        ("White Clipping", f"{(1 - quality.white_clipping_fraction) * 100:.1f}/100"),
    ]


def _table(ax, rows, title=None, font_size=9):
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7)
    table = ax.table(
        cellText=[[a, b] for a, b in rows],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#cbd5e1")
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
    return table


def _comparison_rows(comparison: list[dict[str, Any]]):
    rows = []
    for item in comparison:
        rows.append([
            item.get("model", "unknown"),
            f"{item.get('dice', float('nan')):.4f}",
            f"{item.get('iou', float('nan')):.4f}",
            f"{item.get('precision', float('nan')):.4f}",
            f"{item.get('recall', float('nan')):.4f}",
            f"{item.get('specificity', float('nan')):.4f}",
            f"{item.get('image_f1', float('nan')):.4f}",
        ])
    return rows


def _benchmark_metric_rows(comparison: list[dict[str, Any]]):
    labels = [
        ("Validation Dice", "validation_dice"),
        ("Test Dice", "dice"),
        ("Test IoU", "iou"),
        ("Test HD95", "hd95"),
        ("Test Precision", "precision"),
        ("Test Recall", "recall"),
        ("Test Specificity", "specificity"),
        ("Test MCC", "mcc"),
        ("Image F1", "image_f1"),
        ("Image ROC-AUC", "image_roc_auc"),
        ("Training Epoch", "epoch"),
        ("Training Time (s)", "training_time"),
    ]
    rows = []
    for item in comparison:
        values = [item.get("model", "unknown")]
        for label, key in labels:
            value = item.get(key, float("nan"))
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                if key in {"training_time"}:
                    values.append(f"{float(value):.2f}")
                elif key == "epoch":
                    values.append(f"{int(value)}")
                else:
                    values.append(f"{float(value):.4f}")
            else:
                values.append("—")
        rows.append(values)
    return rows


def _comparison_metrics_are_suspect(comparison: list[dict[str, Any]]) -> bool:
    keys = ("dice", "iou", "precision", "recall", "specificity", "mcc", "image_f1", "image_roc_auc")
    signatures = []
    for item in comparison:
        signature = tuple(round(float(item.get(key, np.nan)), 8) if np.isfinite(float(item.get(key, np.nan))) else None for key in keys)
        signatures.append(signature)
    return len(signatures) >= 3 and len(set(signatures)) == 1


def generate_combined_prediction_report(
    *,
    output_pdf: Path,
    image: np.ndarray,
    quality: ImageQuality,
    image_name: str,
    results: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    threshold: float,
) -> Path:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    display = _display_image(image)
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.965, "Vessel Vision — Multi-Model X-ray Prediction Report", fontsize=16, fontweight="bold", va="top")
        fig.text(0.06, 0.942, f"Image: {image_name}", fontsize=8.5, color="#475569", va="top")
        ax = fig.add_axes([0.16, 0.63, 0.68, 0.28])
        ax.imshow(display, cmap="gray")
        ax.axis("off")
        ax.set_title("Input X-ray", fontsize=11, fontweight="bold", pad=8)
        _table(fig.add_axes([0.06, 0.39, 0.88, 0.18]), _quality_rows(quality), "Image Quality", 8.2)
        summary_rows = []
        for result in results:
            summary_rows.append([
                result.get("model", "—"), result.get("prediction", "—"),
                f"{result.get('mean_pixel_probability', float('nan')):.4f}",
                f"{result.get('peak_pixel_probability', float('nan')):.4f}",
                f"{result.get('predicted_area_percent', float('nan')):.3f}%",
                f"{result.get('inference_time_ms', float('nan')):.2f} ms",
            ])
        ax = fig.add_axes([0.06, 0.12, 0.88, 0.22])
        ax.axis("off")
        ax.set_title(f"Model Outputs (threshold {threshold:.3f})", loc="left", fontsize=11, fontweight="bold", pad=8)
        table = ax.table(cellText=summary_rows, colLabels=["Model", "Prediction", "Mean Prob.", "Peak Prob.", "Area", "Inference"], cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.9])
        table.auto_set_font_size(False); table.set_fontsize(8)
        for cell in table.get_celld().values(): cell.set_edgecolor("#cbd5e1")
        for (r, _c), cell in table.get_celld().items():
            if r == 0: cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
        fig.text(0.06, 0.075, "Research prototype output only. Not a clinical diagnosis.", fontsize=8, color="#475569")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if any(result.get("per_image_metrics") for result in results):
            metric_keys = [
                ("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"),
                ("Recall", "recall"), ("Specificity", "specificity"), ("MCC", "mcc"),
                ("Pixel Accuracy", "pixel_accuracy"), ("FPR", "fpr"), ("FNR", "fnr"), ("HD95", "hd95"),
            ]
            metric_rows = []
            for result in results:
                values = [result.get("model", "—")]
                per_image = result.get("per_image_metrics") or {}
                for _label, key in metric_keys:
                    value = per_image.get(key, np.nan)
                    values.append("—" if not np.isfinite(float(value)) else (f"{float(value):.2f}" if key == "hd95" else f"{float(value):.4f}"))
                metric_rows.append(values)
            fig, ax = plt.subplots(figsize=(12, 6.2)); ax.axis("off")
            ax.set_title("Per-Image Segmentation Metrics — Supplied Ground Truth", loc="left", fontsize=13, fontweight="bold", pad=12)
            table = ax.table(cellText=metric_rows, colLabels=["Model"] + [label for label, _key in metric_keys], cellLoc="center", colLoc="center", bbox=[0, 0.08, 1, 0.84])
            table.auto_set_font_size(False); table.set_fontsize(7.3)
            for cell in table.get_celld().values(): cell.set_edgecolor("#cbd5e1")
            for (r, _c), cell in table.get_celld().items():
                if r == 0: cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
            fig.text(0.02, 0.035, "Per-image scores are computed against the optional ground-truth mask supplied to predict.py.", fontsize=7.5, color="#475569")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, min(4, max(1, len(results))), figsize=(12, 3.5))
        axes = np.atleast_1d(axes)
        for ax, result in zip(axes, results):
            mask_path = Path(result.get("mask", ""))
            if mask_path.is_file():
                mask = np.asarray(plt.imread(mask_path))
                if mask.ndim == 3: mask = mask[..., 0]
            else:
                mask = np.zeros(image.shape, dtype=np.float32)
            ax.imshow(display, cmap="gray")
            ax.imshow(mask, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
            ax.set_title(str(result.get("model", "Model")).upper(), fontsize=9, fontweight="bold")
            ax.axis("off")
        for ax in axes[len(results):]: ax.axis("off")
        fig.suptitle("Model Prediction Outputs", fontsize=14, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        if comparison:
            rows = _benchmark_metric_rows(comparison)
            labels = ["Model", "Val Dice", "Test Dice", "Test IoU", "Test HD95", "Precision", "Recall", "Specificity", "MCC", "Image F1", "ROC-AUC", "Epoch", "Train Time (s)"]
            fig, ax = plt.subplots(figsize=(12, 6.5)); ax.axis("off")
            ax.set_title("Complete Benchmark Metrics — Model Comparison", loc="left", fontsize=13, fontweight="bold", pad=12)
            table = ax.table(cellText=rows, colLabels=labels, cellLoc="center", colLoc="center", bbox=[0, 0.08, 1, 0.84])
            table.auto_set_font_size(False); table.set_fontsize(7.2)
            for cell in table.get_celld().values(): cell.set_edgecolor("#cbd5e1")
            for (r, _c), cell in table.get_celld().items():
                if r == 0: cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
            if _comparison_metrics_are_suspect(comparison):
                fig.text(0.02, 0.035, "WARNING: benchmark metric signatures are identical across all displayed models. Verify experiment isolation before scientific conclusions.", fontsize=7.5, color="#9f1239")
            else:
                fig.text(0.02, 0.035, "Benchmark metrics are model-level reference results, not per-image scores.", fontsize=7.5, color="#475569")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            bounded = [("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"), ("Recall", "recall"), ("Specificity", "specificity"), ("MCC", "mcc"), ("Image F1", "image_f1"), ("ROC-AUC", "image_roc_auc")]
            names = [str(r.get("model", "unknown")) for r in comparison]
            x = np.arange(len(names)); width = 0.1
            fig, ax = plt.subplots(figsize=(12, 6))
            for idx, (label, key) in enumerate(bounded):
                vals = [float(r.get(key, np.nan)) for r in comparison]
                ax.bar(x + (idx - 3.5) * width, vals, width, label=label)
            ax.set_ylim(0, 1); ax.set_xticks(x, names, rotation=20); ax.set_ylabel("Score"); ax.set_title("Model Comparison — Bounded Metrics", fontweight="bold"); ax.grid(axis="y", alpha=0.22); ax.legend(ncol=4, fontsize=8)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            vals = [float(r.get("hd95", np.nan)) for r in comparison]
            ax.bar(names, vals); ax.set_ylabel("Distance (pixels)"); ax.set_title("Model Comparison — HD95 (Lower is Better)", fontweight="bold"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.22)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5)); ax.axis("off")
        ax.set_title("Output Files", loc="left", fontsize=12, fontweight="bold", pad=10)
        file_rows = [[r.get("model", "—"), Path(r.get("mask", "")).name, Path(r.get("overlay", "")).name, Path(r.get("report", "")).name, Path(r.get("json", "")).name] for r in results]
        table = ax.table(cellText=file_rows, colLabels=["Model", "Mask", "Overlay", "PDF Report", "JSON"], cellLoc="left", colLoc="left", bbox=[0, 0.05, 1, 0.82])
        table.auto_set_font_size(False); table.set_fontsize(7.5)
        for cell in table.get_celld().values(): cell.set_edgecolor("#cbd5e1")
        for (r, _c), cell in table.get_celld().items():
            if r == 0: cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
        fig.text(0.02, 0.02, "Per-image segmentation metrics require a ground-truth mask. When provided, they are written to each model JSON result.", fontsize=7.5, color="#475569")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return output_pdf


def generate_prediction_report(
    *,
    output_pdf: Path,
    image: np.ndarray,
    quality: ImageQuality,
    model_name: str,
    image_name: str,
    prediction_mask: np.ndarray,
    probability_map: np.ndarray,
    image_probability: float,
    inference_ms: float,
    threshold: float,
    image_level_fraction: float,
    peak_probability: float | None = None,
    comparison: list[dict[str, Any]] | None = None,
    per_image_metrics: dict[str, Any] | None = None,
) -> Path:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    display = _display_image(image)
    mask = np.asarray(prediction_mask, dtype=bool)
    boundary_mask = boundary(mask)
    area_percent = float(mask.mean() * 100.0)
    prediction = "Positive" if float(mask.mean()) >= image_level_fraction else "Negative"

    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.965, "Vessel Vision — X-ray Quality & AI Prediction Report", fontsize=16, fontweight="bold", va="top")
        fig.text(0.06, 0.942, f"Image: {image_name}", fontsize=8.5, color="#475569", va="top")

        ax_image = fig.add_axes([0.15, 0.60, 0.70, 0.30])
        ax_image.imshow(display, cmap="gray")
        ax_image.axis("off")
        ax_image.set_title("Input X-ray", fontsize=11, pad=8, fontweight="bold")

        _table(fig.add_axes([0.06, 0.43, 0.88, 0.14]), _quality_rows(quality), "Image Quality", 8.5)

        prediction_rows = [
            ("Model", model_name),
            ("Prediction", prediction),
            ("Mean Pixel Probability", f"{image_probability:.4f}"),
            ("Peak Pixel Probability", f"{(peak_probability if peak_probability is not None else np.max(probability_map)):.4f}"),
            ("Predicted Area", f"{area_percent:.3f}%"),
            ("Inference Time", f"{inference_ms:.2f} ms"),
            ("Segmentation Threshold", f"{threshold:.3f}"),
            ("Image-Level Mask Fraction", f"{image_level_fraction:.4f}"),
        ]
        _table(fig.add_axes([0.06, 0.19, 0.88, 0.19]), prediction_rows, "Model Prediction", 8.5)

        if per_image_metrics is not None:
            metric_rows = [(
                label,
                "—" if not np.isfinite(float(per_image_metrics.get(key, np.nan))) else (
                    f"{float(per_image_metrics.get(key)):.4f}" if key != "hd95" else f"{float(per_image_metrics.get(key)):.2f}"
                ),
            ) for label, key in [
                ("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"), ("Recall / Sensitivity", "recall"),
                ("Specificity", "specificity"), ("MCC", "mcc"), ("Pixel Accuracy", "pixel_accuracy"),
                ("FPR", "fpr"), ("FNR", "fnr"), ("HD95", "hd95"),
            ]]
            fig2 = plt.figure(figsize=(8.5, 11)); fig2.patch.set_facecolor("white")
            fig2.text(0.06, 0.965, "Per-Image Ground-Truth Evaluation", fontsize=15, fontweight="bold", va="top")
            fig2.text(0.06, 0.94, "Computed only because a ground-truth mask was supplied.", fontsize=8.5, color="#475569", va="top")
            _table(fig2.add_axes([0.06, 0.55, 0.88, 0.30]), metric_rows, "Segmentation Metrics", 8.5)
            count_rows = [(k.upper(), str(per_image_metrics.get(k, "—"))) for k in ("tp", "tn", "fp", "fn")]
            _table(fig2.add_axes([0.06, 0.28, 0.88, 0.18]), count_rows, "Confusion Counts", 8.5)
            fig2.text(0.06, 0.18, "All values are per-image metrics against the supplied binary ground-truth mask.", fontsize=8, color="#475569")
            fig2.text(0.06, 0.13, "Research prototype output only.", fontsize=9, fontweight="bold")
            fig2.text(0.06, 0.105, "This report is intended for research and evaluation purposes and is not a clinical diagnosis.", fontsize=8, color="#475569")
            pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)

        fig.text(0.06, 0.12, "Research prototype output only.", fontsize=9, fontweight="bold")
        fig.text(0.06, 0.098, "This report is intended for research and evaluation purposes and is not a clinical diagnosis.", fontsize=8, color="#475569")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
        fig.suptitle(f"Segmentation Result — {model_name}", fontsize=14, fontweight="bold")
        axes[0].imshow(display, cmap="gray")
        axes[0].set_title("Original X-ray")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Predicted Mask")
        axes[2].imshow(display, cmap="gray")
        axes[2].imshow(mask, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        if boundary_mask.any():
            axes[2].contour(boundary_mask, levels=[0.5], colors="lime", linewidths=1.2)
        axes[2].set_title("Prediction + Boundary")
        for ax in axes:
            ax.axis("off")
        axes[2].legend(handles=[Patch(facecolor="red", alpha=0.35, label="Prediction"), Patch(facecolor="none", edgecolor="lime", label="Boundary")], loc="lower right", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if comparison:
            fig, ax = plt.subplots(figsize=(12, 6.2)); ax.axis("off")
            ax.set_title("Complete Benchmark Metrics — Model Reference", loc="left", fontsize=13, fontweight="bold", pad=12)
            rows = _benchmark_metric_rows(comparison)
            labels = ["Model", "Val Dice", "Test Dice", "Test IoU", "Test HD95", "Precision", "Recall", "Specificity", "MCC", "Image F1", "ROC-AUC", "Epoch", "Train Time (s)"]
            table = ax.table(cellText=rows, colLabels=labels, cellLoc="center", colLoc="center", bbox=[0, 0.08, 1, 0.84])
            table.auto_set_font_size(False); table.set_fontsize(7.2)
            for cell in table.get_celld().values(): cell.set_edgecolor("#cbd5e1")
            for (r, _c), cell in table.get_celld().items():
                if r == 0: cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
            fig.text(0.02, 0.035, "Benchmark values are model-level reference results, not per-image scores.", fontsize=7.5, color="#475569")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            names = [str(row.get("model", "unknown")) for row in comparison]
            metrics = [
                ("Dice", "dice"),
                ("IoU", "iou"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("Specificity", "specificity"),
                ("Image F1", "image_f1"),
            ]
            x = np.arange(len(names))
            width = 0.12
            fig, ax = plt.subplots(figsize=(11, 5.8))
            for idx, (label, key) in enumerate(metrics):
                vals = [float(row.get(key, np.nan)) for row in comparison]
                ax.bar(x + (idx - 2.5) * width, vals, width, label=label)
            ax.set_ylim(0, 1)
            ax.set_xticks(x, names, rotation=20)
            ax.set_ylabel("Score")
            ax.set_title("Model Comparison — Core Metrics")
            ax.grid(axis="y", alpha=0.22)
            ax.legend(ncol=3, fontsize=8)
            for idx, (label, key) in enumerate(metrics):
                vals = [float(row.get(key, np.nan)) for row in comparison]
                for j, val in enumerate(vals):
                    if np.isfinite(val):
                        ax.text(x[j] + (idx - 2.5) * width, min(val + 0.02, 0.98), f"{val:.2f}", ha="center", va="bottom", fontsize=6)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.axis("off")
            ax.set_title("Benchmark Summary", loc="left", fontsize=13, fontweight="bold", pad=12)
            table = ax.table(
                cellText=_comparison_rows(comparison),
                colLabels=["Model", "Dice", "IoU", "Precision", "Recall", "Specificity", "Image F1"],
                cellLoc="center",
                colLoc="center",
                bbox=[0, 0.15, 1, 0.75],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            for cell in table.get_celld().values():
                cell.set_edgecolor("#cbd5e1")
            for (row, _col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#e2e8f0")
                    cell.set_text_props(weight="bold")
            fig.text(0.02, 0.05, "Comparison values are imported from completed benchmark runs. Repeated or suspect runs should be verified before drawing scientific conclusions.", fontsize=7.5, color="#475569")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_pdf
