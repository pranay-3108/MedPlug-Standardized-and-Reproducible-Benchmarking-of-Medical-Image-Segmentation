"""Visualization and benchmark plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch


class Visualizer:
    @staticmethod
    def save_visualizations(paths, image_tensor, mask_tensor, pred_tensor, threshold=0.5):
        image = Visualizer._prepare_image_array(image_tensor)
        mask = Visualizer._prepare_mask_array(mask_tensor)
        pred = Visualizer._prepare_tensor(pred_tensor).float()
        pred_binary = Visualizer._prepare_mask_array((torch.sigmoid(pred) >= threshold).float())
        Visualizer._save_overlay(paths.overlay_path, image, mask, pred_binary)
        Visualizer._save_mask(paths.prediction_path, pred_binary, "Prediction Mask")
        Visualizer._save_mask(paths.ground_truth_path, mask, "Ground Truth Mask")

    @staticmethod
    def plot_training_history(paths):
        if not paths.training_log_path.exists():
            return
        df = pd.read_csv(paths.training_log_path)
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Epoch"], df["TrainLoss"], marker="o", label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.training_loss_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Epoch"], df["ValidationDiceAll"], marker="o", label="Validation Dice (All)")
        ax.plot(df["Epoch"], df["ValidationDicePositive"], marker="o", label="Validation Dice (Positive)")
        ax.plot(df["Epoch"], df["ValidationIoUAll"], marker="o", label="Validation IoU (All)")
        ax.plot(df["Epoch"], df["ValidationIoUPositive"], marker="o", label="Validation IoU (Positive)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Validation Metrics")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.validation_metrics_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_test_metrics(paths, test_results):
        keys = [
            ("Dice", test_results["all"]["dice"]),
            ("IoU", test_results["all"]["iou"]),
            ("Precision", test_results["all"]["precision"]),
            ("Recall", test_results["all"]["recall"]),
            ("Specificity", test_results["all"]["specificity"]),
            ("MCC", test_results["all"]["mcc"]),
            ("Image F1", test_results["image_classification"]["f1"]),
        ]
        labels = [k for k, _ in keys]
        values = [float(v) if np.isfinite(v) else np.nan for _, v in keys]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values)
        finite = [v for v in values if np.isfinite(v)]
        low = min([-1.0] + finite)
        high = max([1.0] + finite)
        margin = max((high - low) * 0.08, 0.05)
        ax.set_ylim(low - margin, high + margin)
        ax.set_ylabel("Score")
        ax.set_title("Final Test Metrics")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths.test_metrics_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def update_model_comparison(path: Path, benchmark_results_path: Path):
        if not benchmark_results_path.exists():
            return
        df = pd.read_csv(benchmark_results_path)
        if df.empty or "Model" not in df.columns:
            return

        # Keep the newest completed benchmark for each architecture so repeated
        # repair/debug runs do not make the comparison chart unreadable.
        if "Run ID" in df.columns:
            df = df.sort_values("Run ID").drop_duplicates("Model", keep="last")

        metric_columns = [
            ("Test Dice", "Dice"),
            ("Test IoU", "IoU"),
            ("Test Precision", "Precision"),
            ("Test Recall", "Recall"),
            ("Test Specificity", "Specificity"),
            ("Image F1", "Image F1"),
        ]
        available = [(source, label) for source, label in metric_columns if source in df.columns]
        if not available:
            return

        models = df["Model"].astype(str).tolist()
        x = np.arange(len(models))
        width = 0.12
        fig, ax = plt.subplots(figsize=(11, 5.8))
        for index, (source, label) in enumerate(available):
            values = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
            offset = (index - (len(available) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=label)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(float(value) + 0.02, 0.98),
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_ylim(0, 1)
        ax.set_xticks(x, models, rotation=18, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Core Test Metrics", pad=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(ncol=3, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _save_overlay(path, image, mask, pred_binary):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0].set_title("Original X-ray")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[2].imshow(pred_binary, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        edges = Visualizer._boundary(pred_binary)
        if edges.any():
            axes[2].contour(edges, levels=[0.5], colors="lime", linewidths=1.2)
        axes[2].set_title("Prediction Overlay")
        axes[2].axis("off")
        axes[2].legend(
            handles=[
                Patch(facecolor="red", alpha=0.4, label="Prediction"),
                Patch(facecolor="none", edgecolor="lime", label="Boundary"),
            ],
            loc="lower right",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _save_mask(path, mask, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _boundary(mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        padded = np.pad(mask, 1, mode="constant")
        eroded = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[1:-1, 1:-1]
        )
        return mask & ~eroded

    @staticmethod
    def _prepare_tensor(tensor):
        tensor = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    @staticmethod
    def _prepare_image_array(tensor):
        tensor = Visualizer._prepare_tensor(tensor).float()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        return tensor.numpy()

    @staticmethod
    def _prepare_mask_array(tensor):
        tensor = Visualizer._prepare_tensor(tensor).float()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.numpy()
