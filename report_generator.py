"""Research-ready text report generation."""

import ctypes
import os
import platform
from datetime import datetime

import config
import torch


class ReportGenerator:
    def generate(
        self,
        experiment_paths,
        args,
        device,
        dataset_summary,
        validation_results,
        test_results,
        best_epoch,
        stopped_epoch,
        training_time,
        test_time,
        early_stopping_reason,
        benchmark_success=True,
        visualization_status="PASS",
    ):
        report_path = experiment_paths.root / config.REPORT_FILE
        now = datetime.now()
        hardware = self._detect_hardware()
        status = "PASS" if benchmark_success else "FAILED"
        image_cls = test_results["image_classification"]
        lines = [
            "=" * 68,
            "Medical Image Segmentation Benchmark Report",
            "=" * 68,
            "",
            "Experiment Summary",
            "-------------------",
            f"Model: {args.model}",
            "Task: Binary pneumothorax segmentation",
            "Dataset: SIIM-ACR Pneumothorax",
            f"Date: {now.date().isoformat()}",
            f"Device: {device}",
            "",
            "Dataset Summary",
            "---------------",
            f"Train Images: {dataset_summary['train']['loaded']}",
            f"Train Positive: {dataset_summary['train']['positive']}",
            f"Train Negative: {dataset_summary['train']['negative']}",
            f"Validation Images: {dataset_summary['val']['loaded']}",
            f"Validation Positive: {dataset_summary['val']['positive']}",
            f"Validation Negative: {dataset_summary['val']['negative']}",
            f"Test Images: {dataset_summary['test']['loaded']}",
            f"Test Positive: {dataset_summary['test']['positive']}",
            f"Test Negative: {dataset_summary['test']['negative']}",
            f"Configured Image Size: {config.IMAGE_SIZE[0]} x {config.IMAGE_SIZE[1]}",
            "",
            "Hardware",
            "---------",
            f"GPU: {hardware['gpu']}",
            f"GPU VRAM: {hardware['vram']}",
            f"CPU: {hardware['cpu']}",
            f"RAM: {hardware['ram']}",
            "",
            "Training Configuration",
            "----------------------",
            f"Epoch Budget: {args.epochs}",
            f"Batch Size: {args.batch_size}",
            f"Gradient Accumulation: {config.GRADIENT_ACCUMULATION_STEPS}",
            f"Effective Batch Size: {args.batch_size * config.GRADIENT_ACCUMULATION_STEPS}",
            f"Learning Rate: {args.lr}",
            f"Optimizer: {config.OPTIMIZER}",
            f"Weight Decay: {config.WEIGHT_DECAY}",
            f"Scheduler: {config.SCHEDULER}",
            f"Loss Function: {self._loss_name(args.loss)}",
            f"Input Channels: {args.input_channels}",
            f"Normalization: {args.normalization}",
            f"Random Seed: {args.seed}",
            f"Segmentation Threshold: {config.SEGMENTATION_THRESHOLD}",
            f"Image-Level Mask Fraction: {config.IMAGE_LEVEL_MIN_MASK_FRACTION}",
            f"Early Stopping: {'Enabled' if args.early_stopping else 'Disabled'}",
            "",
            "Validation (Best Checkpoint Selection)",
            "---------------------------------------",
            f"Best Validation Dice (All): {validation_results['all']['dice']:.6f}",
            f"Best Validation Dice (Positive): {validation_results['positive']['dice']:.6f}",
            f"Best Validation IoU (All): {validation_results['all']['iou']:.6f}",
            f"Best Validation IoU (Positive): {validation_results['positive']['iou']:.6f}",
            f"Best Epoch: {best_epoch}",
            f"Stopped Epoch: {stopped_epoch}",
            f"Reason: {early_stopping_reason}",
            "",
            "Final Test Results",
            "-------------------",
            f"Selected Metrics: {', '.join(getattr(args, 'metrics', []) or ['dice', 'iou', 'hd95'])}",
        ]
        metric_lines = {
            "dice": f"Test Dice (All): {test_results['all']['dice']:.6f}",
            "positive_dice": f"Test Dice (Positive): {test_results['positive']['dice']:.6f}",
            "iou": f"Test IoU (All): {test_results['all']['iou']:.6f}",
            "positive_iou": f"Test IoU (Positive): {test_results['positive']['iou']:.6f}",
            "hd95": f"HD95 (Positive Cases): {self._fmt(test_results['positive'].get('hd95'))}",
            "precision": f"Precision: {test_results['all']['precision']:.6f}",
            "recall": f"Recall / Sensitivity: {test_results['all']['recall']:.6f}",
            "specificity": f"Specificity: {test_results['all']['specificity']:.6f}",
            "mcc": f"MCC: {test_results['all']['mcc']:.6f}",
            "pixel_accuracy": f"Pixel Accuracy: {test_results['all']['pixel_accuracy']:.6f}",
            "fpr": f"FPR: {test_results['all']['fpr']:.6f}",
            "fnr": f"FNR: {test_results['all']['fnr']:.6f}",
            "image_f1": f"Image F1: {image_cls['f1']:.6f}",
            "image_accuracy": f"Image Accuracy: {image_cls['accuracy']:.6f}",
            "image_precision": f"Image Precision: {image_cls['precision']:.6f}",
            "image_recall": f"Image Recall / Sensitivity: {image_cls['recall']:.6f}",
            "image_specificity": f"Image Specificity: {image_cls['specificity']:.6f}",
            "image_roc_auc": f"Image ROC-AUC: {self._fmt(image_cls['roc_auc'])}",
            "image_pr_auc": f"Image PR-AUC: {self._fmt(image_cls['pr_auc'])}",
        }
        selected = getattr(args, 'metrics', None) or ['dice', 'iou', 'hd95']
        lines.extend(metric_lines[key] for key in selected if key in metric_lines)
        lines.extend([
            "",
            "Image-Level Disease Prediction",
            "--------------------------------",
            f"Selected image-level metrics are shown above; confusion matrix: TP={image_cls['tp']} TN={image_cls['tn']} FP={image_cls['fp']} FN={image_cls['fn']}",
            "",
            "Timing",
            "-------",
            f"Training Time: {training_time:.2f} seconds",
            f"Final Test Time: {test_time:.2f} seconds",
            "",
            "Generated Files",
            "---------------",
            experiment_paths.config_path.name,
            experiment_paths.metrics_path.name,
            experiment_paths.training_log_path.name,
            experiment_paths.training_loss_plot_path.name,
            experiment_paths.validation_metrics_plot_path.name,
            experiment_paths.test_metrics_plot_path.name,
            experiment_paths.model_comparison_path.name,
            experiment_paths.overlay_path.name,
            experiment_paths.prediction_path.name,
            experiment_paths.ground_truth_path.name,
            f"{experiment_paths.checkpoint_path.parent.name}/{experiment_paths.checkpoint_path.name}",
            f"{experiment_paths.resume_checkpoint_path.parent.name}/{experiment_paths.resume_checkpoint_path.name}",
            "",
            "Benchmark Status",
            "----------------",
            status,
            f"Visualization Status: {visualization_status}",
            "",
            "Interpretation",
            "--------------",
            "The experiment used the configured fixed train/validation/test protocol. "
            "The validation set was used for checkpoint selection, and the test set was evaluated only once after training.",
            "",
            "Experiment Directory",
            "--------------------",
            str(experiment_paths.root.resolve()),
        ])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    @staticmethod
    def _fmt(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "Not available"
        return "Not available" if value != value else f"{value:.6f}"

    @staticmethod
    def _loss_name(loss_name):
        return "BCEWithLogitsLoss" if loss_name == "bce" else f"DiceBCELoss (bce_weight={config.BCE_WEIGHT})"

    @staticmethod
    def _detect_hardware():
        vram = "Not available"
        gpu = "CPU Only"
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        return {"gpu": gpu, "vram": vram, "cpu": platform.processor() or platform.machine() or "Unknown", "ram": ReportGenerator._ram_description()}

    @staticmethod
    def _ram_description():
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return f"{status.total_physical / (1024 ** 3):.2f} GB"
        return "Unavailable"
