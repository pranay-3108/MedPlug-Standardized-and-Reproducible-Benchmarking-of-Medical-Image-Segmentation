"""Experiment artifact management for isolated, reproducible benchmark runs."""

import csv
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import config


_RUN_ID_RE = re.compile(r"^RUN_\d{8}_\d{6}_[A-Z0-9]{5}$")


@dataclass(frozen=True)
class ExperimentPaths:
    run_id: str
    root: Path
    manifest_path: Path
    config_path: Path
    metrics_path: Path
    training_log_path: Path
    checkpoint_path: Path
    resume_checkpoint_path: Path
    overlay_path: Path
    prediction_path: Path
    ground_truth_path: Path
    training_loss_plot_path: Path
    validation_metrics_plot_path: Path
    test_metrics_plot_path: Path
    model_comparison_path: Path


class ExperimentManager:
    """Create and track one isolated artifact tree per experiment run."""

    def __init__(self, experiments_root=None):
        self.experiments_root = (
            Path(experiments_root)
            if experiments_root is not None
            else config.EXPERIMENTS_DIRECTORY
        )
        self.experiments_root.mkdir(parents=True, exist_ok=True)

        # These remain shared derived indexes for now. The run directory is
        # always the source of truth for an individual experiment.
        self.benchmark_results_path = self.experiments_root / config.BENCHMARK_RESULTS_FILE
        self.leaderboard_path = self.experiments_root / config.LEADERBOARD_FILE

    @staticmethod
    def new_run_id():
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"RUN_{now}_{uuid.uuid4().hex[:5].upper()}"

    def create_experiment(self, model_name, checkpoint_name=None, resume=False, run_id=None, existing=False):
        if resume:
            if not run_id:
                raise ValueError("Resume requested but no run_id was provided.")
            run_root = self._run_root(run_id)
            if not run_root.is_dir():
                raise FileNotFoundError(f"Cannot resume unknown run: {run_id}")
        else:
            run_id = run_id or self.new_run_id()
            run_root = self._run_root(run_id)
            if run_root.exists() and not existing:
                raise FileExistsError(f"Run directory already exists: {run_root}")

        # Resolve the effective checkpoint name ONCE, from model_name, before
        # building any paths. Previously _build_paths() received the raw
        # (possibly None) checkpoint_name and fell back internally to the
        # global config.MODEL_NAME default -- not this run's model_name.
        # That meant a run created for e.g. "segformer" could get an
        # ExperimentPaths object stamped with "unet"-derived filenames
        # (visible as a stray "unet_latest_training.pth" entry in
        # manifest["artifacts"] for non-default models), even though the
        # manifest's own "checkpoint_name" field was computed correctly.
        # Resolving it here keeps every path derived from paths, and the
        # manifest field, in lockstep.
        resolved_checkpoint_name = checkpoint_name or f"{model_name}_best_model"
        paths = self._build_paths(run_id, run_root, resolved_checkpoint_name)

        for directory in (
            paths.root,
            paths.root / "config",
            paths.root / "logs",
            paths.root / "checkpoints",
            paths.root / "metrics",
            paths.root / "predictions",
            paths.root / "visualizations",
            paths.root / "comparisons",
            paths.root / "report",
            paths.root / "audit",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if not resume and not existing:
            self._write_json(
                paths.manifest_path,
                {
                    "run_id": run_id,
                    "status": "CREATED",
                    "stage": "CREATED",
                    "created_at": self._now(),
                    "started_at": None,
                    "completed_at": None,
                    "model": model_name,
                    "checkpoint_name": resolved_checkpoint_name,
                    "artifacts": {},
                    "error": None,
                },
            )
        return paths

    def mark_started(self, paths, args=None, effective_device=None):
        manifest = self.load_manifest(paths)
        manifest.update(
            {
                "status": "RUNNING",
                "stage": "TRAINING",
                "started_at": manifest.get("started_at") or self._now(),
            }
        )
        if args is not None:
            manifest["model"] = args.model
            manifest["configuration"] = self._args_snapshot(args)
        if effective_device is not None:
            manifest["device"] = str(effective_device)
        self._write_json(paths.manifest_path, manifest)

    def mark_completed(self, paths, stage="COMPLETED"):
        manifest = self.load_manifest(paths)
        manifest.update(
            {
                "status": "COMPLETED",
                "stage": stage,
                "completed_at": self._now(),
                "error": None,
            }
        )
        manifest["artifacts"] = self.collect_artifacts(paths)
        self._write_json(paths.manifest_path, manifest)

    def mark_failed(self, paths, error, stage=None):
        manifest = self.load_manifest(paths)
        manifest.update(
            {
                "status": "FAILED",
                "stage": stage or manifest.get("stage") or "UNKNOWN",
                "error": str(error),
            }
        )
        manifest["artifacts"] = self.collect_artifacts(paths)
        self._write_json(paths.manifest_path, manifest)

    def update_stage(self, paths, stage, status="RUNNING", error=None):
        manifest = self.load_manifest(paths)
        manifest["stage"] = stage
        manifest["status"] = status
        manifest["error"] = error
        manifest["artifacts"] = self.collect_artifacts(paths)
        self._write_json(paths.manifest_path, manifest)

    def load_manifest(self, paths):
        if not paths.manifest_path.exists():
            raise FileNotFoundError(f"Run manifest missing: {paths.manifest_path}")
        return json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    def list_runs(self):
        runs = []
        for path in sorted(self.experiments_root.glob("RUN_*")):
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append(manifest)
        return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_paths(self, run_id):
        run_root = self._run_root(run_id)
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._build_paths(
            run_id,
            run_root,
            manifest.get("checkpoint_name") or f"{manifest.get('model', config.MODEL)}_best_model",
        )

    def initialize_artifacts(self, paths, args, effective_device, reset=True):
        now = datetime.now(timezone.utc)
        values = [
            ("Model", args.model),
            ("Task", "Binary pneumothorax segmentation"),
            ("Dataset Root", getattr(args, "data_root", config.PROJECT_ROOT)),
            ("Train CSV", getattr(args, "data_root", config.PROJECT_ROOT) / getattr(args, "train_csv_rel", config.TRAIN_CSV.name)),
            ("Validation CSV", getattr(args, "data_root", config.PROJECT_ROOT) / getattr(args, "val_csv_rel", config.VAL_CSV.name)),
            ("Test CSV", getattr(args, "data_root", config.PROJECT_ROOT) / getattr(args, "test_csv_rel", config.TEST_CSV.name)),
            ("Train DICOM Directory", getattr(args, "data_root", config.PROJECT_ROOT) / getattr(args, "train_images_rel", "images/dicom-images-train")),
            ("Test DICOM Directory", getattr(args, "data_root", config.PROJECT_ROOT) / getattr(args, "test_images_rel", "images/dicom-images-test")),
            ("Image Size", config.IMAGE_SIZE),
            ("Input Channels", args.input_channels),
            ("Output Channels", args.output_channels),
            ("Batch Size", args.batch_size),
            ("Gradient Accumulation Steps", config.GRADIENT_ACCUMULATION_STEPS),
            ("Effective Batch Size", args.batch_size * config.GRADIENT_ACCUMULATION_STEPS),
            ("Epochs", args.epochs),
            ("Learning Rate", args.lr),
            ("Weight Decay", config.WEIGHT_DECAY),
            ("Optimizer", config.OPTIMIZER),
            ("Scheduler", config.SCHEDULER),
            ("Loss", self._loss_name(args.loss)),
            ("Normalization", args.normalization),
            ("Segmentation Threshold", config.SEGMENTATION_THRESHOLD),
            ("Image Level Mask Fraction", config.IMAGE_LEVEL_MIN_MASK_FRACTION),
            ("Early Stopping", args.early_stopping),
            ("Patience", args.patience),
            ("Device", effective_device),
            ("Random Seed", args.seed),
            ("Resume", getattr(args, "resume", False)),
            ("Checkpoint Name", args.checkpoint_name),
            ("Run ID", paths.run_id),
            ("Date", now.date().isoformat()),
            ("Time UTC", now.time().isoformat(timespec="seconds")),
            ("Selected Metrics", ", ".join(getattr(args, "metrics", [])) or "dice, iou, hd95"),
        ]
        if reset or not paths.config_path.exists():
            self._atomic_write_text(
                paths.config_path,
                "\n".join(f"{k}={v}" for k, v in values) + "\n",
            )
        if reset or not paths.training_log_path.exists():
            self._write_csv_header(
                paths.training_log_path,
                [
                    "Epoch",
                    "TrainLoss",
                    "ValidationDiceAll",
                    "ValidationDicePositive",
                    "ValidationIoUAll",
                    "ValidationIoUPositive",
                    "LearningRate",
                    "ElapsedSeconds",
                ],
            )
        if reset or not paths.metrics_path.exists():
            self._write_csv_header(paths.metrics_path, ["Metric", "Value"])

        manifest = self.load_manifest(paths)
        manifest["configuration"] = self._args_snapshot(args)
        manifest["device"] = str(effective_device)
        manifest["model"] = args.model
        manifest["checkpoint_name"] = args.checkpoint_name
        manifest["status"] = "RUNNING"
        manifest["stage"] = "TRAINING"
        manifest["started_at"] = manifest.get("started_at") or self._now()
        self._write_json(paths.manifest_path, manifest)

        self._initialize_benchmark_results()
        self._initialize_leaderboard()

    def _initialize_benchmark_results(self):
        columns = [
            "Run ID", "Date", "Model", "Validation Dice", "Test Dice", "Test IoU",
            "Test HD95", "Test Precision", "Test Recall", "Test Specificity", "Test MCC",
            "Image F1", "Image ROC-AUC", "Epoch", "Training Time", "Device", "Checkpoint Name",
            "Protocol", "Validation Dice (Positive)", "Test Dice (Positive)", "Test IoU (Positive)",
            "Test Pixel Accuracy", "Test FPR", "Test FNR", "Image Accuracy", "Image Precision",
            "Image Recall", "Image Specificity", "Image PR-AUC", "Verification Status",
            "Metric Source", "Image Metrics Status", "Test Count", "Positive Test Count",
            "Validation/Test Overlap", "Audit Mismatches",
        ]
        if not self.benchmark_results_path.exists() or self.benchmark_results_path.stat().st_size == 0:
            self._write_csv_header(self.benchmark_results_path, columns)
            return
        existing = self._read_csv_columns(self.benchmark_results_path)
        if existing == columns:
            return
        if existing and "Run ID" not in existing:
            self._migrate_csv_with_run_id(self.benchmark_results_path, columns, "Model")
            return
        rows = []
        with self.benchmark_results_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self._atomic_write_csv(self.benchmark_results_path, rows, columns)

    def _initialize_leaderboard(self):
        columns = [
            "Run ID", "Model", "Protocol", "Dice", "Positive Dice", "IoU", "Positive IoU", "HD95",
            "Precision", "Recall", "Specificity", "MCC", "Pixel Accuracy", "FPR", "FNR",
            "Image F1", "Image ROC-AUC", "Image PR-AUC", "Training Time", "Best Epoch", "Date",
            "Device", "Verification Status", "Image Metrics Status", "Test Count",
            "Positive Test Count", "Validation/Test Overlap", "Metric Source", "Audit Mismatches",
        ]
        if not self.leaderboard_path.exists() or self.leaderboard_path.stat().st_size == 0:
            self._write_csv_header(self.leaderboard_path, columns)
            return
        existing = self._read_csv_columns(self.leaderboard_path)
        if existing == columns:
            return
        if existing and "Run ID" not in existing:
            self._migrate_csv_with_run_id(self.leaderboard_path, columns, "Model")
            return
        rows = []
        with self.leaderboard_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self._atomic_write_csv(self.leaderboard_path, rows, columns)

    def append_training_log(self, paths, epoch, train_loss, validation_results, learning_rate, elapsed_seconds):
        with paths.training_log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch,
                train_loss,
                validation_results["all"]["dice"],
                validation_results["positive"]["dice"],
                validation_results["all"]["iou"],
                validation_results["positive"]["iou"],
                learning_rate,
                elapsed_seconds,
            ])

    def save_metrics(
        self,
        paths,
        best_validation_dice,
        best_validation_iou,
        validation_results,
        test_results,
        training_time,
        args,
        device,
        best_epoch,
        stopped_epoch,
        early_stopping_reason,
    ):
        metric_values = {
            "dice": ("Test Dice (All)", test_results["all"]["dice"]),
            "positive_dice": ("Test Dice (Positive)", test_results["positive"]["dice"]),
            "iou": ("Test IoU (All)", test_results["all"]["iou"]),
            "positive_iou": ("Test IoU (Positive)", test_results["positive"]["iou"]),
            "hd95": ("Test HD95", test_results["positive"].get("hd95", float("nan"))),
            "precision": ("Test Precision", test_results["all"]["precision"]),
            "recall": ("Test Recall", test_results["all"]["recall"]),
            "specificity": ("Test Specificity", test_results["all"]["specificity"]),
            "mcc": ("Test MCC", test_results["all"]["mcc"]),
            "pixel_accuracy": ("Test Pixel Accuracy", test_results["all"]["pixel_accuracy"]),
            "fpr": ("Test FPR", test_results["all"]["fpr"]),
            "fnr": ("Test FNR", test_results["all"]["fnr"]),
            "image_f1": ("Image F1", test_results["image_classification"]["f1"]),
            "image_accuracy": ("Image Accuracy", test_results["image_classification"]["accuracy"]),
            "image_precision": ("Image Precision", test_results["image_classification"]["precision"]),
            "image_recall": ("Image Recall", test_results["image_classification"]["recall"]),
            "image_specificity": ("Image Specificity", test_results["image_classification"]["specificity"]),
            "image_roc_auc": ("Image ROC-AUC", test_results["image_classification"]["roc_auc"]),
            "image_pr_auc": ("Image PR-AUC", test_results["image_classification"]["pr_auc"]),
        }
        selected = getattr(args, "metrics", None) or ["dice", "iou", "hd95"]
        selected_text = ", ".join(selected)
        rows = [
            ("Selected Metrics", selected_text),
            ("Best Validation Dice (Positive)", best_validation_dice),
            ("Best Validation IoU (Positive)", best_validation_iou),
            ("Training Time", training_time),
            ("Best Epoch", best_epoch),
            ("Stopped Epoch", stopped_epoch),
            ("Early Stopping", "Enabled" if args.early_stopping else "Disabled"),
            ("Early Stopping Reason", early_stopping_reason),
        ]
        # Persist every available metric. The selection in the UI is metadata for
        # the run, not permission to drop core evaluation results from the record.
        rows += list(metric_values.values())

        # Keep the run-local artifact schema identical to its validator.
        self._write_csv_rows(
            paths.metrics_path,
            rows,
            mode="w",
            columns=["Metric", "Value"],
        )

        # The run-local metrics file is authoritative. These CSVs are derived
        # indexes and are deliberately kept outside the run directory.
        self.append_benchmark_results(paths, test_results, training_time, args, device, best_epoch)
        self.update_leaderboard(paths, test_results, training_time, args, device, best_epoch)

        if not self._valid_metrics_file(paths.metrics_path):
            raise RuntimeError(f"Metrics artifact was not committed correctly: {paths.metrics_path}")

    def append_benchmark_results(self, paths, test_results, training_time, args, device, best_epoch):
        self._initialize_benchmark_results()
        now = datetime.now(timezone.utc)
        def fmt(value):
            try:
                number = float(value)
                return f"{number:.10g}" if math.isfinite(number) else ""
            except (TypeError, ValueError, OverflowError):
                return ""

        image_labels = test_results.get("image_classification", {})
        path_hint = " ".join(str(getattr(args, key, "")) for key in (
            "source_mode", "train_csv_rel", "val_csv_rel", "test_csv_rel", "train_csv", "val_csv", "test_csv", "config"
        )).lower()
        if "diagnostic" in path_hint or "aligned" in path_hint:
            protocol = "B (positive-only diagnostic)"
        elif "test-rle" in path_hint or "official" in path_hint:
            protocol = "A (official split)"
        else:
            protocol = "inspect run manifest/config"
        row = {
            "Run ID": paths.run_id,
            "Date": now.date().isoformat(),
            "Model": args.model,
            "Validation Dice": fmt(test_results.get("validation_dice")),
            "Test Dice": fmt(test_results["all"].get("dice")),
            "Test IoU": fmt(test_results["all"].get("iou")),
            "Test HD95": fmt(test_results["positive"].get("hd95")),
            "Test Precision": fmt(test_results["all"].get("precision")),
            "Test Recall": fmt(test_results["all"].get("recall")),
            "Test Specificity": fmt(test_results["all"].get("specificity")),
            "Test MCC": fmt(test_results["all"].get("mcc")),
            "Image F1": fmt(image_labels.get("f1")),
            "Image ROC-AUC": fmt(image_labels.get("roc_auc")),
            "Epoch": best_epoch,
            "Training Time": fmt(training_time),
            "Device": str(device),
            "Checkpoint Name": args.checkpoint_name,
            "Protocol": protocol,
            "Validation Dice (Positive)": fmt(test_results.get("validation_positive_dice")),
            "Test Dice (Positive)": fmt(test_results.get("positive", {}).get("dice")),
            "Test IoU (Positive)": fmt(test_results.get("positive", {}).get("iou")),
            "Test Pixel Accuracy": fmt(test_results["all"].get("pixel_accuracy")),
            "Test FPR": fmt(test_results["all"].get("fpr")),
            "Test FNR": fmt(test_results["all"].get("fnr")),
            "Image Accuracy": fmt(image_labels.get("accuracy")),
            "Image Precision": fmt(image_labels.get("precision")),
            "Image Recall": fmt(image_labels.get("recall")),
            "Image Specificity": fmt(image_labels.get("specificity")),
            "Image PR-AUC": fmt(image_labels.get("pr_auc")),
            "Verification Status": "PENDING_AUDIT",
            "Metric Source": "run-local evaluation; audit pending",
            "Image Metrics Status": "CHECK_TEST_CLASS_BALANCE",
            "Test Count": "",
            "Positive Test Count": "",
            "Validation/Test Overlap": "NOT_CHECKED",
            "Audit Mismatches": "",
        }
        columns = [
            "Run ID", "Date", "Model", "Validation Dice", "Test Dice", "Test IoU", "Test HD95",
            "Test Precision", "Test Recall", "Test Specificity", "Test MCC", "Image F1",
            "Image ROC-AUC", "Epoch", "Training Time", "Device", "Checkpoint Name", "Protocol",
            "Validation Dice (Positive)", "Test Dice (Positive)", "Test IoU (Positive)",
            "Test Pixel Accuracy", "Test FPR", "Test FNR", "Image Accuracy", "Image Precision",
            "Image Recall", "Image Specificity", "Image PR-AUC", "Verification Status", "Metric Source",
            "Image Metrics Status", "Test Count", "Positive Test Count", "Validation/Test Overlap", "Audit Mismatches",
        ]
        if self.benchmark_results_path.exists():
            existing_columns = self._read_csv_columns(self.benchmark_results_path)
            if existing_columns and "Run ID" not in existing_columns:
                self._migrate_csv_with_run_id(
                    self.benchmark_results_path,
                    columns,
                    "Model",
                )
        rows = []
        if self.benchmark_results_path.exists():
            with self.benchmark_results_path.open("r", newline="", encoding="utf-8") as handle:
                rows = [item for item in csv.DictReader(handle) if item.get("Run ID") != paths.run_id]
        rows.append(row)
        self._atomic_write_csv(self.benchmark_results_path, rows, columns)

    def update_leaderboard(self, paths, test_results, training_time, args, device, best_epoch):
        self._initialize_leaderboard()
        columns = [
            "Run ID", "Model", "Protocol", "Dice", "Positive Dice", "IoU", "Positive IoU", "HD95",
            "Precision", "Recall", "Specificity", "MCC", "Pixel Accuracy", "FPR", "FNR",
            "Image F1", "Image ROC-AUC", "Image PR-AUC", "Training Time", "Best Epoch", "Date",
            "Device", "Verification Status", "Image Metrics Status", "Test Count",
            "Positive Test Count", "Validation/Test Overlap", "Metric Source", "Audit Mismatches",
        ]
        if self.leaderboard_path.exists():
            existing_columns = self._read_csv_columns(self.leaderboard_path)
            if existing_columns and "Run ID" not in existing_columns:
                self._migrate_csv_with_run_id(self.leaderboard_path, columns, "Model")

        def fmt(value):
            try:
                number = float(value)
                return f"{number:.10g}" if math.isfinite(number) else ""
            except (TypeError, ValueError, OverflowError):
                return ""

        image = test_results.get("image_classification", {})
        path_hint = " ".join(str(getattr(args, key, "")) for key in (
            "source_mode", "train_csv_rel", "val_csv_rel", "test_csv_rel", "train_csv", "val_csv", "test_csv", "config"
        )).lower()
        if "diagnostic" in path_hint or "aligned" in path_hint:
            protocol = "B (positive-only diagnostic)"
        elif "test-rle" in path_hint or "official" in path_hint:
            protocol = "A (official split)"
        else:
            protocol = "inspect run manifest/config"
        row = {
            "Run ID": paths.run_id,
            "Model": args.model,
            "Protocol": protocol,
            "Dice": fmt(test_results["all"].get("dice")),
            "Positive Dice": fmt(test_results["positive"].get("dice")),
            "IoU": fmt(test_results["all"].get("iou")),
            "Positive IoU": fmt(test_results["positive"].get("iou")),
            "HD95": fmt(test_results["positive"].get("hd95")),
            "Precision": fmt(test_results["all"].get("precision")),
            "Recall": fmt(test_results["all"].get("recall")),
            "Specificity": fmt(test_results["all"].get("specificity")),
            "MCC": fmt(test_results["all"].get("mcc")),
            "Pixel Accuracy": fmt(test_results["all"].get("pixel_accuracy")),
            "FPR": fmt(test_results["all"].get("fpr")),
            "FNR": fmt(test_results["all"].get("fnr")),
            "Image F1": fmt(image.get("f1")),
            "Image ROC-AUC": fmt(image.get("roc_auc")),
            "Image PR-AUC": fmt(image.get("pr_auc")),
            "Training Time": fmt(training_time),
            "Best Epoch": best_epoch,
            "Date": datetime.now(timezone.utc).date().isoformat(),
            "Device": str(device),
            "Verification Status": "PENDING_AUDIT",
            "Image Metrics Status": "CHECK_TEST_CLASS_BALANCE",
            "Test Count": "",
            "Positive Test Count": "",
            "Validation/Test Overlap": "NOT_CHECKED",
            "Metric Source": "run-local evaluation; audit pending",
            "Audit Mismatches": "",
        }

        # Phase 1 does not add a database/queue. Keep the existing leaderboard
        # as a derived index, now with explicit run identity.
        rows = []
        if self.leaderboard_path.exists():
            with self.leaderboard_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        rows.append(row)
        rows.sort(key=lambda value: self._float_or_neg_inf(value.get("Dice")), reverse=True)
        self._atomic_write_csv(self.leaderboard_path, rows, columns)

    def collect_artifacts(self, paths):
        candidates = {
            "config": paths.config_path,
            "metrics": paths.metrics_path,
            "training_log": paths.training_log_path,
            "checkpoint": paths.checkpoint_path,
            "resume_checkpoint": paths.resume_checkpoint_path,
            "overlay": paths.overlay_path,
            "prediction": paths.prediction_path,
            "ground_truth": paths.ground_truth_path,
            "training_loss_plot": paths.training_loss_plot_path,
            "validation_metrics_plot": paths.validation_metrics_plot_path,
            "test_metrics_plot": paths.test_metrics_plot_path,
            "model_comparison": paths.model_comparison_path,
            "report": paths.root / config.REPORT_FILE,
            "report_pdf": paths.root / config.REPORT_PDF_FILE,
        }
        audit_dir = paths.root / "audit"
        if audit_dir.is_dir():
            for audit_path in audit_dir.rglob("*"):
                if audit_path.is_file():
                    rel = audit_path.relative_to(paths.root).as_posix()
                    candidates.setdefault(rel, audit_path)
        return {
            key: str(path.relative_to(paths.root))
            for key, path in candidates.items()
            if path.is_file()
        }

    def _build_paths(self, run_id, run_root, checkpoint_name):
        # checkpoint_name should always be resolved by the caller (using this
        # run's own model_name/manifest, e.g. in create_experiment/get_paths)
        # before reaching here. The config.MODEL_NAME fallback below is only
        # a last-resort safety net for legacy/unexpected callers -- relying
        # on it previously caused non-default-model runs to be silently
        # stamped with paths for the global default model instead of their
        # own. Prefer fixing the caller over depending on this fallback.
        checkpoint_stem = Path(checkpoint_name or f"{config.MODEL_NAME}_best_model").stem
        return ExperimentPaths(
            run_id=run_id,
            root=run_root,
            manifest_path=run_root / "manifest.json",
            config_path=run_root / "config" / config.CONFIG_FILE,
            metrics_path=run_root / "metrics" / config.METRICS_FILE,
            training_log_path=run_root / "logs" / config.TRAINING_LOG_FILE,
            checkpoint_path=run_root / "checkpoints" / f"{checkpoint_stem}.pth",
            resume_checkpoint_path=run_root / "checkpoints" / f"{checkpoint_stem}_latest_training.pth",
            overlay_path=run_root / "visualizations" / config.OVERLAY_FILE,
            prediction_path=run_root / "visualizations" / config.PREDICTION_FILE,
            ground_truth_path=run_root / "visualizations" / config.GROUND_TRUTH_FILE,
            training_loss_plot_path=run_root / "visualizations" / config.TRAINING_LOSS_PLOT_FILE,
            validation_metrics_plot_path=run_root / "visualizations" / config.VALIDATION_METRICS_PLOT_FILE,
            test_metrics_plot_path=run_root / "visualizations" / config.TEST_METRICS_PLOT_FILE,
            model_comparison_path=run_root / "comparisons" / config.MODEL_COMPARISON_FILE,
        )

    def _run_root(self, run_id):
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run_id: {run_id}")
        return self.experiments_root / run_id

    @staticmethod
    def _args_snapshot(args):
        snapshot = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                snapshot[key] = str(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                snapshot[key] = value
            elif isinstance(value, (list, tuple)):
                snapshot[key] = [str(item) if isinstance(item, Path) else item for item in value]
            else:
                snapshot[key] = str(value)

        # Record the resolved dataset paths actually used by train.py instead
        # of stale global config defaults. The validated SIIM-ACR mapping uses
        # Record the exact image source configuration actually used.
        data_root = snapshot.get("data_root")
        train_rel = snapshot.get("train_images_rel") or "images/dicom-images-train"
        val_rel = snapshot.get("val_images_rel") or train_rel
        test_rel = snapshot.get("test_images_rel") or "images/dicom-images-test"
        aligned_images_rel = snapshot.get("aligned_images_rel") or "aligned_dataset/images"
        aligned_masks_rel = snapshot.get("aligned_masks_rel") or "aligned_dataset/masks"
        snapshot["source_mode"] = snapshot.get("source_mode") or "auto"
        if data_root:
            root = Path(data_root)
            snapshot["train_dicom_dir"] = str((root / str(train_rel)).resolve())
            snapshot["val_dicom_dir"] = str((root / str(val_rel)).resolve())
            snapshot["test_dicom_dir"] = str((root / str(test_rel)).resolve())
            snapshot["aligned_images_dir"] = str((root / str(aligned_images_rel)).resolve())
            snapshot["aligned_masks_dir"] = str((root / str(aligned_masks_rel)).resolve())

        return snapshot

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()


    @staticmethod
    def atomic_torch_save(obj, path):
        """Write a torch artifact completely before replacing the final path."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(obj, temp_path)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _atomic_write_text(path, text):
        ExperimentManager._atomic_write_bytes(path, text.encode("utf-8"))

    @staticmethod
    def _write_json(path, data):
        payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
        ExperimentManager._atomic_write_text(path, payload)

    @staticmethod
    def _atomic_write_bytes(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_bytes(data)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _write_csv_rows(path, rows, mode="w", columns=None):
        path = Path(path)
        if mode == "w":
            temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temp_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if columns:
                        writer.writerow(columns)
                    writer.writerows(rows)
                os.replace(temp_path, path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            return

        with path.open(mode, newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

    @staticmethod
    def _write_csv_header(path, columns):
        ExperimentManager._atomic_write_text(path, ",".join(columns) + "\n")

    @staticmethod
    def _append_csv_row(path, row, columns):
        path = Path(path)
        exists = path.exists() and path.stat().st_size > 0
        # Phase 1 avoids a global DB; append is kept small and source-of-truth
        # remains run-local. This method is intentionally narrow for now.
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _atomic_write_csv(path, rows, columns):
        path = Path(path)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _read_csv_columns(path):
        try:
            with Path(path).open("r", newline="", encoding="utf-8") as handle:
                return next(csv.reader(handle), [])
        except (OSError, StopIteration):
            return []

    def _migrate_csv_with_run_id(self, path, columns, model_column):
        existing = []
        try:
            with Path(path).open("r", newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        except OSError:
            existing = []
        migrated = []
        for index, row in enumerate(existing):
            row = dict(row)
            row["Run ID"] = row.get("Run ID") or f"LEGACY_{index + 1:05d}"
            migrated.append({column: row.get(column, "") for column in columns})
        self._atomic_write_csv(path, migrated, columns)

    @staticmethod
    def _valid_metrics_file(path):
        try:
            with Path(path).open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            return len(rows) >= 2 and rows[0][:2] == ["Metric", "Value"]
        except (OSError, UnicodeError):
            return False

    @staticmethod
    def _float_or_neg_inf(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _loss_name(loss_name):
        return "BCEWithLogitsLoss" if loss_name == "bce" else "DiceBCELoss (bce_weight=0.5)"

    @staticmethod
    def start_timer():
        return perf_counter()

    @staticmethod
    def elapsed_seconds(start_time):
        return perf_counter() - start_time