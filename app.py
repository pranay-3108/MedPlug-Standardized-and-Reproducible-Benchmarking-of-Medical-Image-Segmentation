"""Local MedPlugBench control panel for isolated experiment runs.

Phase 1 goals:
- every submission receives a unique run_id
- configuration/logs/artifacts are stored inside that run
- previous runs are never deleted
- the UI reads run history from persistent manifests
"""

from __future__ import annotations

import csv
import html
from email.parser import BytesParser
from email.policy import default as email_default_policy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode

import config
from diagnostics import (
    artifact_health,
    architecture_audit,
    derived_error_profile,
    explainability_readiness,
    metric_warnings,
    render_diagnostics_html,
    training_stability,
)
from experiment_manager import ExperimentManager, ExperimentPaths
from research_figures import render_research_figures


ROOT = config.PROJECT_ROOT.resolve()
PORT = 8765
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_ALLOWED_IMAGE_SUFFIXES = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp"}
_MODELS = ("unet", "attention_unet", "transunet", "segformer")
_LOSSES = ("dice_bce", "bce")
_SCHEDULERS = ("cosine", "none")
_DEVICES = ("auto", "cuda", "cpu")
_NORMALIZATION = ("minmax", "imagenet")
_METRICS = (
    ("dice", "Dice (all)"),
    ("positive_dice", "Dice (positive)"),
    ("iou", "IoU (all)"),
    ("positive_iou", "IoU (positive)"),
    ("hd95", "HD95 (positive)"),
    ("precision", "Precision"),
    ("recall", "Recall / Sensitivity"),
    ("specificity", "Specificity"),
    ("mcc", "MCC"),
    ("pixel_accuracy", "Pixel Accuracy"),
    ("fpr", "FPR"),
    ("fnr", "FNR"),
    ("image_f1", "Image F1"),
    ("image_accuracy", "Image Accuracy"),
    ("image_precision", "Image Precision"),
    ("image_recall", "Image Recall"),
    ("image_specificity", "Image Specificity"),
    ("image_roc_auc", "Image ROC-AUC"),
    ("image_pr_auc", "Image PR-AUC"),
)

_MANAGER = ExperimentManager()
_TRAIN_PROCESS: subprocess.Popen[str] | None = None
_TRAIN_RUN_ID: str | None = None
_TRAIN_LOG_HANDLE = None
_TRAIN_LOCK = threading.Lock()


def _options(values, selected):
    return "".join(
        f'<option value="{html.escape(value)}"{" selected" if value == selected else ""}>{html.escape(value)}</option>'
        for value in values
    )


def _field(label, name, value, kind="text", min_value=None, step=None):
    attrs = [f'name="{html.escape(name)}"', f'value="{html.escape(str(value))}"']
    if kind == "number" and min_value is not None:
        attrs.append(f'min="{html.escape(str(min_value))}"')
    if step is not None:
        attrs.append(f'step="{html.escape(str(step))}"')
    return f'<label>{html.escape(label)}<input type="{kind}" {" ".join(attrs)}></label>'


def _checkbox(name, label, checked=False):
    flag = " checked" if checked else ""
    return f'<label class="check"><input type="checkbox" name="{html.escape(name)}" value="1"{flag}>{html.escape(label)}</label>'


def _yaml_quote(value):
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _parse_int(form, key, default, minimum=0):
    try:
        value = int(form.getfirst(key, str(default)))
    except ValueError:
        return default
    return max(value, minimum)


def _parse_float(form, key, default, minimum=None):
    try:
        value = float(form.getfirst(key, str(default)))
    except ValueError:
        return default
    return max(value, minimum) if minimum is not None else value


def _selected_metrics(form):
    selected = [key for key, _ in _METRICS if key in form]
    return selected or [key for key, _ in _METRICS]


def _has_dataset_layout(root: Path) -> bool:
    root = Path(root).expanduser()
    csvs = (
        root / "data" / "train-rle.csv",
        root / "data" / "val-rle.csv",
        root / "data" / "test-rle.csv",
    )
    if not all(path.is_file() for path in csvs):
        return False
    raw_dicom = root / "images" / "dicom-images-train"
    aligned = root / "aligned_dataset" / "images"
    masks = root / "aligned_dataset" / "masks"
    return raw_dicom.is_dir() or (aligned.is_dir() and masks.is_dir())


def _dataset_candidates() -> list[Path]:
    home = Path.home()
    candidates = []

    env_root = os.environ.get("MEDPLUG_DATASET_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend([
        ROOT / "nothing",
        ROOT / "dataset",
        ROOT.parent / "nothing",
        ROOT.parent / "dataset",
        Path("C:/MedPlugBench/nothing"),
        Path("C:/MedPlugBench/dataset"),
    ])

    # The UI often lives under Downloads while the large dataset is kept in
    # a sibling folder. Only inspect one directory level; do not scan a whole drive.
    for base in (home / "Downloads", home / "Documents", home / "Desktop"):
        if not base.is_dir():
            continue
        candidates.append(base)
        try:
            candidates.extend(child for child in base.iterdir() if child.is_dir())
        except OSError:
            pass

    unique = []
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.expanduser().resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _default_dataset_root() -> Path | None:
    for candidate in _dataset_candidates():
        if _has_dataset_layout(candidate):
            return candidate.resolve()
    return None


def _normalize_dataset_root(value: str) -> str:
    raw = value.strip()
    if not raw:
        default = _default_dataset_root()
        return str(default) if default else ""

    path = Path(raw).expanduser()
    if _has_dataset_layout(path):
        return str(path.resolve())

    # Handle a common copy/paste typo such as ...\MedPlugBench\MedPlugBench\nothing.
    parts = list(path.parts)
    for index in range(len(parts) - 1):
        if parts[index].lower() == "medplugbench" and parts[index + 1].lower() == "medplugbench":
            fixed = Path(*parts[: index + 1], *parts[index + 2 :])
            if _has_dataset_layout(fixed):
                return str(fixed.resolve())

    return str(path.resolve())


def _state_path(paths: ExperimentPaths) -> Path:
    return paths.root / "process_state.json"


def _write_process_state(paths: ExperimentPaths, status: str, **extra) -> None:
    payload = {"run_id": paths.run_id, "status": status, "updated_at": time.time(), **extra}
    state_path = _state_path(paths)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_process_state(run_id):
    if not run_id:
        return {}
    try:
        path = _get_run_paths(run_id).root / "process_state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        return {}


def _get_run_paths(run_id):
    # _MANAGER.get_paths() is the single source of truth for where a run's
    # artifacts live: it re-derives every path from that run's own
    # manifest.json (model + checkpoint_name), so it can never drift onto a
    # different model's filenames. The block this replaced was a flat,
    # global-config-based fallback (e.g. a shared config.CHECKPOINT_DIRECTORY
    # and a single global config.RESUME_CHECKPOINT_NAME reused for every
    # model) that does not match how ExperimentManager actually lays out a
    # run's directory tree, and would silently report artifacts as missing.
    # It is intentionally removed rather than kept "just in case" -- a
    # fallback that produces wrong paths is worse than a clear error.
    if not hasattr(_MANAGER, "get_paths"):
        raise RuntimeError(
            "Experiment manager does not implement get_paths(); cannot "
            "safely resolve run artifact locations."
        )
    return _MANAGER.get_paths(run_id)



def _manifest_artifact_paths(run_id: str) -> dict[str, Path]:
    """Resolve run artifacts from the persisted manifest first."""
    manifest = _run_manifest(run_id)
    run_root = _MANAGER._run_root(run_id)
    mapping = {}
    for key, rel in (manifest.get("artifacts") or {}).items():
        if not rel:
            continue
        path = run_root / str(rel).replace("\\", "/")
        # Old manifests can contain stale artifact names from before the
        # per-model checkpoint isolation fix. Ignore missing manifest entries
        # so diagnostics can fall back to the run's current expected paths.
        if path.is_file():
            mapping[key] = path
    return mapping


def _diagnostic_path_map(run_id: str) -> dict[str, Path]:
    """Build diagnostic paths from manifest artifact records, with safe fallbacks."""
    manifest_paths = _manifest_artifact_paths(run_id)
    raw = _get_run_paths(run_id)
    defaults = {
        "checkpoint": raw.checkpoint_path,
        "metrics": raw.metrics_path,
        "training_log": raw.training_log_path,
        "training_loss_plot": raw.training_loss_plot_path,
        "validation_metrics_plot": raw.validation_metrics_plot_path,
        "test_metrics_plot": raw.test_metrics_plot_path,
        "prediction": raw.prediction_path,
        "ground_truth": raw.ground_truth_path,
        "overlay": raw.overlay_path,
        "model_comparison": raw.model_comparison_path,
        "report": raw.root / config.REPORT_FILE,
        "metric_status": raw.root / "audit" / "metric_status.json",
        "metric_ledger": raw.root / "audit" / "metric_ledger.csv",
        "test_predictions": raw.root / "audit" / "test_predictions.csv",
        "validation_predictions": raw.root / "audit" / "validation_predictions.csv",
        "verification": raw.root / "audit" / "verification.json",
        "run_forensics": raw.root / "audit" / "run_forensics.json",
        "checkpoint_hashes": raw.root / "audit" / "checkpoint_hashes.json",
        "checkpoint_metadata": raw.root / "audit" / "best_checkpoint_metadata.json",
        "model_signature": raw.root / "audit" / "model_signature.json",
        "dataset_signature": raw.root / "audit" / "dataset_signature.json",
        "training_history_json": raw.root / "audit" / "training_history.json",
        "architecture": raw.root / "audit" / "architecture.txt",
        "run_command": raw.root / "audit" / "run_command.txt",
        "dataset_stats": raw.root / "audit" / "dataset_stats.json",
        "prediction_behavior_graph": raw.root / "audit" / "prediction_behavior.png",
        "learning_rate_graph": raw.root / "audit" / "learning_rate_curve.png",
        "probability_graph": raw.root / "audit" / "probability_stats.png",
        "logit_graph": raw.root / "audit" / "logit_stats.png",
        "gradient_graph": raw.root / "audit" / "gradient_stability.png",
        "confusion_graph": raw.root / "audit" / "test_confusion_matrix.png",
    }
    paths = {}
    for key, fallback in defaults.items():
        paths[key] = manifest_paths.get(key, fallback)
    paths["_run_root"] = raw.root
    return paths


def _list_runs():
    if hasattr(_MANAGER, "list_runs"):
        return _MANAGER.list_runs()
    runs = []
    root = config.EXPERIMENTS_DIRECTORY
    if not root.exists():
        return runs
    for run_dir in sorted(root.glob("RUN_*"), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs.append(manifest)
        except (OSError, ValueError, TypeError):
            continue
    return runs


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, ProcessLookupError):
        return False


def _discover_active_run():
    for run in _list_runs():
        run_id = str(run.get("run_id", ""))
        state = _read_process_state(run_id)
        if state.get("status") in {"STARTING", "RUNNING"} and _pid_alive(state.get("pid")):
            return run_id
    return None


def _powershell_executable() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"


def _write_ui_config(form, paths: ExperimentPaths):
    """Write one self-contained YAML config for the prepared run."""
    dataset_root = _normalize_dataset_root(
        form.getfirst("dataset_root", str(_default_dataset_root()))
    )

    model = form.getfirst("model", config.MODEL)
    if model not in _MODELS:
        model = config.MODEL
    loss = form.getfirst("loss", "dice_bce")
    if loss not in _LOSSES:
        loss = "dice_bce"
    scheduler = form.getfirst("scheduler", "cosine")
    if scheduler not in _SCHEDULERS:
        scheduler = "cosine"
    device = form.getfirst("device", "auto")
    if device not in _DEVICES:
        device = "auto"
    normalization = form.getfirst("normalization", "minmax")
    if normalization not in _NORMALIZATION:
        normalization = "minmax"

    image_height = _parse_int(form, "image_height", 256, 32)
    image_width = _parse_int(form, "image_width", 256, 32)
    metrics = _selected_metrics(form)

    lines = [
        "project:",
        "  name: medplugbench_ui_experiment",
        "",
        "dataset:",
        "  plugin: siim_acr",
        f"  root: {_yaml_quote(dataset_root)}",
        '  train_csv: "data/train-rle.csv"',
        '  val_csv: "data/val-rle.csv"',
        '  test_csv: "data/test-rle.csv"',
        '  train_images: "images/dicom-images-train"',
        '  val_images: "images/dicom-images-train"',
        '  test_images: "images/dicom-images-test"',
        '  aligned_images: "aligned_dataset/images"',
        '  aligned_masks: "aligned_dataset/masks"',
        '  source_mode: "auto"',
        "",
        "preprocess:",
        "  plugin: segmentation_default",
        f"  image_size: [{image_height}, {image_width}]",
        f"  normalization: {normalization}",
        "  input_channels: 1",
        "",
        "model:",
        f"  plugin: {model}",
        f'  pretrained: {str("pretrained" in form).lower()}',
        "",
        "training:",
        f"  batch_size: {_parse_int(form, 'batch_size', 2, 1)}",
        f"  num_workers: {_parse_int(form, 'num_workers', 0, 0)}",
        f"  epochs: {_parse_int(form, 'epochs', 25, 1)}",
        f"  learning_rate: {_parse_float(form, 'learning_rate', 0.0001, 0.0)}",
        f"  weight_decay: {_parse_float(form, 'weight_decay', 0.0001, 0.0)}",
        f"  optimizer: {_yaml_quote(form.getfirst('optimizer', 'adamw'))}",
        f"  scheduler: {scheduler}",
        f"  gradient_accumulation_steps: {_parse_int(form, 'gradient_accumulation', 2, 1)}",
        f"  random_seed: {_parse_int(form, 'seed', 42, 0)}",
        f"  device: {device}",
        f"  loss: {loss}",
        f"  early_stopping_enabled: {str('early_stopping' in form).lower()}",
        f"  patience: {_parse_int(form, 'patience', 10, 1)}",
        f"  segmentation_threshold: {_parse_float(form, 'threshold', 0.50, 0.0)}",
        "  resume: false",
        "",
        "evaluation:",
        "  metrics:",
    ]
    lines.extend(f"    - {metric}" for metric in metrics)

    config_path = paths.config_path.parent / "experiment.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _run_preflight(config_path):
    command = [
        sys.executable,
        str(ROOT / "preflight.py"),
        "--config",
        str(Path(config_path).resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "Preflight timed out. Training was not started.", ""

    output = (result.stdout or "") + (result.stderr or "")
    return (
        result.returncode == 0,
        "Preflight passed: READY FOR TRAINING."
        if result.returncode == 0
        else "Preflight failed. Training was NOT started.",
        output,
    )


def _pipeline_command(paths, config_path):
    python = Path(sys.executable).resolve()
    runner = (ROOT / "run_pipeline.py").resolve()
    config_path = Path(config_path).resolve()
    return (
        f'cd "{ROOT}"\n'
        f'& "{python}" "{runner}" --config "{config_path}" --run-id "{paths.run_id}"'
    )


def _prepare_run(paths, config_path):
    command = _pipeline_command(paths, config_path)
    command_path = paths.root / "logs" / "training_command.txt"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(command + "\n", encoding="utf-8")
    _write_process_state(paths, "READY", stage="READY", command=command)
    return command


def _training_status():
    active_run_id = _discover_active_run()
    if active_run_id:
        return f"RUNNING — {active_run_id}", None, active_run_id
    return "No training job owned by app.py. Prepared runs continue independently in PowerShell.", None, None


def _run_manifest(run_id):
    try:
        return _MANAGER.load_manifest(_get_run_paths(run_id))
    except (OSError, ValueError, TypeError):
        return None


def _log_tail(run_id, lines=100):
    if not run_id:
        return "No run selected."
    paths = _get_run_paths(run_id)
    candidates = [
        paths.root / "logs" / "pipeline.log",
        paths.root / "logs" / "training_log.csv",
        paths.root / "logs" / "powershell_training.log",
        paths.root / "logs" / "ui_train.log",
        paths.root / "logs" / "training.log",
    ]
    existing = [path for path in candidates if path.exists() and path.is_file()]
    if existing:
        log_path = max(existing, key=lambda path: path.stat().st_mtime)
        if log_path.name == "training_log.csv":
            history = _load_training_history(run_id)
            recent = history[-min(lines, 20):]
            if recent:
                return "\n".join(
                    f"Epoch {row.get('Epoch')}: train_loss={row.get('TrainLoss')} val_dice+={row.get('ValidationDicePositive')} val_iou+={row.get('ValidationIoUPositive')} lr={row.get('LearningRate')}"
                    for row in recent
                )
        content = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])
    return "No training log yet."


def _metric_checks(selected=None):
    selected = set(selected or [key for key, _ in _METRICS])
    checks = "".join(_checkbox(key, label, key in selected) for key, label in _METRICS)
    return (
        '<div class="metric-actions">'
        '<button type="button" onclick="setAllMetrics(true)">Select All</button>'
        '<button type="button" onclick="setAllMetrics(false)">Clear All</button>'
        '</div>'
        f'<div id="metric-list">{checks}</div>'
    )



def _escape(value):
    return html.escape(str(value))


def _load_csv_dicts(path):
    if not path or not Path(path).is_file():
        return []
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []


def _load_test_metrics(run_id):
    """Load a complete metric view without hiding audit findings.

    Run-local metrics.csv contains the subset chosen in older UI versions.
    The audit ledger fills gaps, and independent recomputation takes precedence
    for numerical values. Verification status is displayed separately so a
    recomputed number is never silently treated as a fully verified run.
    """
    if not run_id:
        return {}

    try:
        paths = _diagnostic_path_map(run_id)
    except (OSError, ValueError, TypeError, KeyError):
        paths = {}

    result = {}

    def add_csv_metrics(path):
        for row in _load_csv_dicts(path):
            name = str(row.get("Metric", "")).strip()
            if not name or name == "Selected Metrics":
                continue
            try:
                value = float(row.get("Value", ""))
                if math.isfinite(value):
                    result[name] = value
            except (TypeError, ValueError, OverflowError):
                continue

    import math
    add_csv_metrics(paths.get("metrics"))

    # Metric ledger includes metrics which old metrics.csv files omitted.
    ledger_path = paths.get("metric_ledger")
    for row in _load_csv_dicts(ledger_path):
        name = str(row.get("Metric", "")).strip()
        status = str(row.get("Status", "")).upper()
        if not name or status not in {"OK", "COMPUTED", "RECOMPUTED"}:
            continue
        try:
            value = float(row.get("Value", ""))
            if math.isfinite(value):
                result.setdefault(name, value)
        except (TypeError, ValueError, OverflowError):
            continue

    # Independent reconstruction is the best available numerical source.
    # Keep verification status/mismatches separately visible in the UI.
    verification_path = paths.get("verification")
    if verification_path and Path(verification_path).is_file():
        try:
            verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            verification = {}
        recomputed = verification.get("recomputed", {}) if isinstance(verification, dict) else {}
        if isinstance(recomputed, dict):
            for name in (
                "Test Dice (All)", "Test Dice (Positive)", "Test IoU (All)",
                "Test IoU (Positive)", "Test HD95", "Test Precision", "Test Recall",
                "Test Specificity", "Test MCC", "Test Pixel Accuracy", "Test FPR",
                "Test FNR", "Image Accuracy", "Image Precision", "Image Recall",
                "Image Specificity", "Image F1", "Image ROC-AUC", "Image PR-AUC",
            ):
                raw = recomputed.get(name)
                try:
                    value = float(raw)
                    if math.isfinite(value):
                        result[name] = value
                    elif name in result:
                        result.pop(name, None)
                except (TypeError, ValueError, OverflowError):
                    # An explicit undefined value (e.g. ROC-AUC for one class)
                    # must not be replaced by a stale reported number.
                    if name in recomputed and raw is None:
                        result.pop(name, None)

    # Old archives may not contain the per-run artifacts. Fall back to the
    # shared benchmark index only for values still absent from the sources above.
    benchmark_rows = _load_csv_dicts(config.EXPERIMENTS_DIRECTORY / config.BENCHMARK_RESULTS_FILE)
    index_mapping = {
        "Validation Dice": "Validation Dice (All)",
        "Validation Dice (Positive)": "Validation Dice (Positive)",
        "Test Dice": "Test Dice (All)",
        "Test Dice (Positive)": "Test Dice (Positive)",
        "Test IoU": "Test IoU (All)",
        "Test IoU (Positive)": "Test IoU (Positive)",
        "Test HD95": "Test HD95",
        "Test Precision": "Test Precision",
        "Test Recall": "Test Recall",
        "Test Specificity": "Test Specificity",
        "Test MCC": "Test MCC",
        "Test Pixel Accuracy": "Test Pixel Accuracy",
        "Test FPR": "Test FPR",
        "Test FNR": "Test FNR",
        "Image Accuracy": "Image Accuracy",
        "Image Precision": "Image Precision",
        "Image Recall": "Image Recall",
        "Image Specificity": "Image Specificity",
        "Image F1": "Image F1",
        "Image ROC-AUC": "Image ROC-AUC",
        "Image PR-AUC": "Image PR-AUC",
    }
    for row in benchmark_rows:
        if str(row.get("Run ID", "")).strip() != str(run_id):
            continue
        for source, target in index_mapping.items():
            if target in result:
                continue
            try:
                value = float(row.get(source, ""))
                if math.isfinite(value):
                    result[target] = value
            except (TypeError, ValueError, OverflowError):
                continue
        break
    return result


def _metric_audit_note(run_id):
    """Explain verification state and class-balance caveats near metric views."""
    if not run_id:
        return ""
    try:
        paths = _diagnostic_path_map(run_id)
    except (OSError, ValueError, TypeError, KeyError):
        return ""
    verification_path = paths.get("verification")
    verification = {}
    if verification_path and Path(verification_path).is_file():
        try:
            verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            verification = {}

    status = str(verification.get("status", "NOT_AVAILABLE")).upper()
    mismatches = verification.get("mismatches", []) or []
    if status == "PASS":
        heading = "Independent metric verification: PASS"
    elif status == "FAILED":
        heading = "Independent metric verification: FAILED — review the mismatches before citing the run."
    else:
        heading = "Independent metric verification: NOT AVAILABLE — saved results have not been independently recomputed."

    details = []
    for row in mismatches[:5]:
        details.append(
            f"{row.get('metric', 'metric')}: reported {row.get('reported', '—')}, "
            f"recomputed {row.get('recomputed', '—')}"
        )
    try:
        test_rows = _load_csv_dicts(paths.get("test_predictions"))
    except (OSError, ValueError, TypeError):
        test_rows = []
    if test_rows:
        positives = sum(1 for row in test_rows if str(row.get("GroundTruthPositive", "0")).strip() == "1")
        if positives == len(test_rows):
            details.append(
                "Image-level ROC-AUC is undefined because the test set has no negative cases; "
                "image-level PR-AUC is not informative on this positive-only set."
            )

    detail_html = "<br>".join(_escape(item) for item in details)
    return (
        '<section class="card"><span class="eyebrow">METRIC INTEGRITY</span>'
        f'<h3>{_escape(heading)}</h3>'
        + (f'<p class="muted">{detail_html}</p>' if detail_html else "")
        + '</section>'
    )


def _load_training_history(run_id):
    if not run_id:
        return []
    try:
        path = _diagnostic_path_map(run_id).get("training_log")
    except (OSError, ValueError, TypeError):
        return []
    return _load_csv_dicts(path)


def _run_metric_value(run_id, metric_key, split="test"):
    if split == "test":
        mapping = {
            "dice": "Test Dice (All)",
            "positive_dice": "Test Dice (Positive)",
            "iou": "Test IoU (All)",
            "positive_iou": "Test IoU (Positive)",
            "hd95": "Test HD95",
            "precision": "Test Precision",
            "recall": "Test Recall",
            "specificity": "Test Specificity",
            "mcc": "Test MCC",
            "pixel_accuracy": "Test Pixel Accuracy",
            "fpr": "Test FPR",
            "fnr": "Test FNR",
            "image_f1": "Image F1",
            "image_accuracy": "Image Accuracy",
            "image_precision": "Image Precision",
            "image_recall": "Image Recall",
            "image_specificity": "Image Specificity",
            "image_roc_auc": "Image ROC-AUC",
            "image_pr_auc": "Image PR-AUC",
        }
        label = mapping.get(metric_key)
        return _load_test_metrics(run_id).get(label) if label else None

    validation_map = {
        "dice": "ValidationDiceAll",
        "positive_dice": "ValidationDicePositive",
        "iou": "ValidationIoUAll",
        "positive_iou": "ValidationIoUPositive",
    }
    column = validation_map.get(metric_key)
    if not column:
        return None
    values = []
    for row in _load_training_history(run_id):
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(values) if values else None


def _metric_label(metric_key):
    return dict(_METRICS).get(metric_key, metric_key.replace("_", " ").title())


def _metric_direction(metric_key):
    return "lower" if metric_key in {"hd95", "fpr", "fnr"} else "higher"


def _metric_options(split, selected="recall"):
    validation_keys = {"dice", "positive_dice", "iou", "positive_iou"}
    allowed = validation_keys if split == "validation" else {key for key, _ in _METRICS}
    return "".join(
        f'<option value="{_escape(key)}"{" selected" if key == selected else ""}>{_escape(label)}</option>'
        for key, label in _METRICS if key in allowed
    )


def _completed_runs():
    return [
        run for run in _list_runs()
        if str(run.get("status", "")).upper() in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
    ]


def _prediction_runs():
    """Completed runs that have a real checkpoint available for inference."""
    runs = []
    for run in _completed_runs():
        model = str(run.get("model", "")).strip()
        run_id = str(run.get("run_id", "")).strip()
        if not model or not run_id:
            continue
        try:
            paths = _get_run_paths(run_id)
        except (OSError, ValueError, KeyError):
            continue
        candidates = [paths.checkpoint_path, ROOT / "checkpoints" / f"{model}_best_model.pth"]
        artifact = (run.get("artifacts") or {}).get("checkpoint")
        if artifact:
            candidates.append(paths.root / str(artifact).replace("\\", "/"))
        if any(Path(candidate).is_file() for candidate in candidates):
            runs.append(run)
    return runs


def _run_config(run):
    return run.get("configuration", {}) if isinstance(run, dict) else {}


def _compatible_runs(selected_run):
    selected_cfg = _run_config(selected_run)
    dataset_plugin = selected_cfg.get("dataset_plugin") or "siim_acr"
    data_root = str(selected_cfg.get("data_root", "")).rstrip("\\/").lower()
    compatible = []
    for run in _completed_runs():
        cfg = _run_config(run)
        if (cfg.get("dataset_plugin") or "siim_acr") != dataset_plugin:
            continue
        other_root = str(cfg.get("data_root", "")).rstrip("\\/").lower()
        if data_root and other_root and data_root != other_root:
            continue
        compatible.append(run)
    return compatible


def _svg_bar_chart(title, values, direction="higher"):
    if not values:
        return '<div class="empty">No comparable values available.</div>'
    width, height = 760, 330
    left, top, chart_w, chart_h = 70, 55, 650, 205
    nums = [value for _, value in values]
    lo, hi = min(nums), max(nums)
    if lo == hi:
        pad = max(abs(hi) * 0.1, 1.0)
        lo -= pad
        hi += pad
    elif 0.0 <= lo and hi <= 1.0:
        lo, hi = 0.0, 1.0
    else:
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad
    bar_w = chart_w / len(values) * 0.62
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="16" font-weight="700">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#94a3b8"/>',
    ]
    for i in range(5):
        y_value = lo + (hi - lo) * i / 4
        y = top + chart_h - chart_h * (y_value - lo) / (hi - lo)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{y_value:.3g}</text>')
    for i, (label, value) in enumerate(values):
        x = left + (i + 0.5) * (chart_w / len(values)) - bar_w / 2
        bar_h = chart_h * (value - lo) / (hi - lo)
        y = top + chart_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="#2563eb"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{max(y - 7, top + 12):.1f}" text-anchor="middle" font-size="11" font-weight="700">{value:.4g}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + chart_h + 18}" text-anchor="middle" font-size="11" fill="#334155">{_escape(label[:18])}</text>')
    direction_text = "higher is better" if direction == "higher" else "lower is better"
    parts.append(f'<text x="{left + chart_w}" y="{height - 10}" text-anchor="end" font-size="11" fill="#64748b">{direction_text}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _svg_grouped_metric_chart(title, run_ids, metric_keys, split="test"):
    completed = {str(run.get("run_id")): run for run in _completed_runs()}
    runs = [completed[rid] for rid in run_ids if rid in completed]
    if not runs or not metric_keys:
        return '<div class="empty">No comparable values available.</div>'
    width, height = 900, 410
    left, top, chart_w, chart_h = 75, 60, 760, 255
    group_w = chart_w / len(runs)
    bar_w = min(22, group_w / max(len(metric_keys) + 2, 1))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#94a3b8"/>',
    ]
    for i in range(6):
        y_value = i / 5
        y = top + chart_h - chart_h * y_value
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{y_value:.1f}</text>')
    labels = [_metric_label(key).replace("Test ", "") for key in metric_keys]
    for run_index, run in enumerate(runs):
        center = left + group_w * run_index + group_w / 2
        start_x = center - (len(metric_keys) * bar_w) / 2
        for metric_index, key in enumerate(metric_keys):
            value = _run_metric_value(str(run.get("run_id")), key, split)
            if value is None:
                continue
            value = max(0.0, min(1.0, float(value)))
            x = start_x + metric_index * bar_w
            bar_h = chart_h * value
            y = top + chart_h - bar_h
            opacity = 0.55 + 0.07 * metric_index
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 5):.1f}" height="{max(bar_h, 1):.1f}" rx="3" fill="#2563eb" opacity="{opacity:.2f}"/>')
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{max(y - 4, top + 9):.1f}" text-anchor="middle" font-size="7" fill="#334155">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + chart_h + 22}" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{_escape(str(run.get("model", "unknown"))[:17])}</text>')
    for index, label in enumerate(labels):
        x = left + index * 125
        opacity = 0.55 + 0.07 * index
        parts.append(f'<rect x="{x}" y="{height - 42}" width="10" height="10" rx="2" fill="#2563eb" opacity="{opacity:.2f}"/>')
        parts.append(f'<text x="{x + 15}" y="{height - 33}" font-size="9" fill="#475569">{_escape(label[:17])}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _metric_table(run_ids, metric_keys, split="test"):
    rows = []
    completed = {str(run.get("run_id")): run for run in _completed_runs()}
    for run_id in run_ids:
        run = completed.get(run_id)
        if not run:
            continue
        label = f"{run.get('model', 'unknown')} — {run_id}"
        values = [_run_metric_value(run_id, key, split) for key in metric_keys]
        cells = "".join(f'<td>{"—" if value is None else f"{value:.6f}"}</td>' for value in values)
        rows.append(f'<tr><td><strong>{_escape(label)}</strong></td>{cells}</tr>')
    headers = "".join(f'<th>{_escape(_metric_label(key))}</th>' for key in metric_keys)
    if not rows:
        return '<div class="empty">No compatible completed runs have the requested metrics.</div>'
    return f'<div class="table-scroll"><table class="data-table"><thead><tr><th>Run</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _kpi_cards(run_id):
    metrics = _load_test_metrics(run_id)
    selected = [
        ("Test Dice", metrics.get("Test Dice (All)")),
        ("Positive Dice", metrics.get("Test Dice (Positive)")),
        ("Test IoU", metrics.get("Test IoU (All)")),
        ("Recall", metrics.get("Test Recall")),
        ("Specificity", metrics.get("Test Specificity")),
        ("MCC", metrics.get("Test MCC")),
        ("PR-AUC", metrics.get("Image PR-AUC")),
    ]
    cards = []
    for name, value in selected:
        text = "—" if value is None else f"{value:.4f}"
        cards.append(f'<div class="kpi"><div class="kpi-label">{_escape(name)}</div><div class="kpi-value">{text}</div></div>')
    return "".join(cards)


def _artifact_browser(paths):
    items = [
        ("Checkpoint", paths.checkpoint_path),
        ("Resume checkpoint", paths.resume_checkpoint_path),
        ("Metrics CSV", paths.metrics_path),
        ("Training log", paths.training_log_path),
        ("Pipeline log", paths.root / "logs" / "pipeline.log"),
        ("Prediction overlay", paths.overlay_path),
        ("Prediction mask", paths.prediction_path),
        ("Ground truth", paths.ground_truth_path),
        ("Training curve", paths.training_loss_plot_path),
        ("Validation curve", paths.validation_metrics_plot_path),
        ("Test metrics chart", paths.test_metrics_plot_path),
        ("Model comparison", paths.model_comparison_path),
        ("Report (text)", paths.root / config.REPORT_FILE),
        ("Report (PDF)", paths.root / config.REPORT_PDF_FILE),
    ]
    rows = []
    for label, path in items:
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            rows.append(f'<tr><td>{_escape(label)}</td><td>{_escape(path.name)}</td><td><a href="/{quote(rel)}" target="_blank">Open</a></td></tr>')
    if not rows:
        return '<p class="muted">No artifacts available yet.</p>'
    return '<table class="data-table"><thead><tr><th>Artifact</th><th>File</th><th></th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'


def _image_card(run_id, path, title):
    if not run_id or not path.is_file():
        return f'<section class="card"><h3>{_escape(title)}</h3><p class="muted">Not generated.</p></section>'
    relative = path.relative_to(ROOT).as_posix()
    return f'<section class="card"><h3>{_escape(title)}</h3><img src="/{quote(relative)}" alt="{_escape(title)}" loading="lazy"></section>'


def _runs_table(selected_run_id=None):
    runs = _list_runs()
    if not runs:
        return '<p class="muted">No experiments have been created yet.</p>'
    rows = []
    for run in runs:
        run_id = str(run.get("run_id", ""))
        selected = " selected" if run_id == selected_run_id else ""
        rows.append(f'<option value="{_escape(run_id)}"{selected}>{_escape(run_id)} — {_escape(str(run.get("model", "")))} — {_escape(str(run.get("status", "")))}</option>')
    return '<form class="inline-form" method="get" action="/"><input type="hidden" name="view" value="dashboard"><label>Open run<select name="run_id">' + "".join(rows) + '</select></label><button type="submit">Open Run</button></form>'


def _nav(view, run_id=None):
    parts = []
    for key, label in [("dashboard", "Dashboard"), ("setup", "Experiment Setup"), ("analysis", "Metric Explorer"), ("compare", "Compare Models"), ("research", "Research Figures"), ("prediction", "Prediction"), ("diagnostics", "Diagnostics")]:
        query = {"view": key}
        if run_id:
            query["run_id"] = run_id
        href = "/?" + urlencode(query)
        parts.append(f'<a class="nav-link {"active" if view == key else ""}" href="{href}">{label}</a>')
    return "<nav>" + "".join(parts) + "</nav>"


def _setup_section(selected_model):
    return f'''<section class="card"><h2>Experiment Setup</h2>
<p class="muted">Configure a run, verify the validated dataset contract, then prepare an independent PowerShell command. app.py does not own the long-running training process.</p>
<form method="post" action="/run"><div class="form-grid">
{_field("Dataset root", "dataset_root", str(_default_dataset_root()) if _default_dataset_root() else "")}
{_field("Batch size", "batch_size", 2, "number", 1)}
{_field("Epochs", "epochs", 25, "number", 1)}
{_field("Learning rate", "learning_rate", 0.0001, "number", 0, "any")}
{_field("Weight decay", "weight_decay", 0.0001, "number", 0, "any")}
{_field("Gradient accumulation", "gradient_accumulation", 2, "number", 1)}
{_field("Workers", "num_workers", 0, "number", 0)}
{_field("Seed", "seed", 42, "number", 0)}
{_field("Patience", "patience", 10, "number", 1)}
{_field("Segmentation threshold", "threshold", 0.50, "number", 0, "any")}
{_field("Image height", "image_height", 256, "number", 32)}
{_field("Image width", "image_width", 256, "number", 32)}
<label>Model<select name="model">{_options(_MODELS, selected_model)}</select></label>
<label>Loss<select name="loss">{_options(_LOSSES, "dice_bce")}</select></label>
<label>Optimizer<select name="optimizer"><option value="adamw" selected>adamw</option></select></label>
<label>Scheduler<select name="scheduler">{_options(_SCHEDULERS, "cosine")}</select></label>
<label>Device<select name="device">{_options(_DEVICES, "auto")}</select></label>
<label>Normalization<select name="normalization">{_options(_NORMALIZATION, "minmax")}</select></label>
</div><div class="checks-row">{_checkbox("pretrained", "Use pretrained model weights", False)}{_checkbox("early_stopping", "Enable early stopping", False)}</div>
<h3>Metrics to include</h3>{_metric_checks()}<button type="submit">Verify &amp; Prepare PowerShell Run</button></form></section>'''


def _render_analysis_form(run_id, split="test", metric="recall"):
    compatible = _compatible_runs(_run_manifest(run_id) or {})
    values = []
    for run in compatible:
        rid = str(run.get("run_id"))
        value = _run_metric_value(rid, metric, split)
        if value is not None:
            values.append((str(run.get("model", "unknown")), value))
    chart = _svg_bar_chart(f"{_metric_label(metric)} — {split.title()}", values, _metric_direction(metric))
    history = _load_training_history(run_id)
    history_table = ""
    if history:
        recent = history[-5:]
        history_table = '<table class="data-table"><thead><tr><th>Epoch</th><th>Train Loss</th><th>Val Dice+</th><th>Val IoU+</th><th>LR</th></tr></thead><tbody>' + "".join(
            f'<tr><td>{_escape(row.get("Epoch"))}</td><td>{_escape(row.get("TrainLoss"))}</td><td>{_escape(row.get("ValidationDicePositive"))}</td><td>{_escape(row.get("ValidationIoUPositive"))}</td><td>{_escape(row.get("LearningRate"))}</td></tr>'
            for row in recent
        ) + '</tbody></table>'
    all_metric_table = _metric_table([run_id], list(dict(_METRICS).keys()), "test") if split == "test" else _metric_table([run_id], ["dice", "positive_dice", "iou", "positive_iou"], "validation")
    return f'''<section class="card"><h2>Metric Explorer</h2>
<p class="muted">Pick one metric and compare compatible completed runs. Test metrics come from metrics.csv; validation metrics come from the saved training history.</p>
<form class="explore-form" method="get" action="/"><input type="hidden" name="view" value="analysis"><input type="hidden" name="run_id" value="{_escape(run_id)}">
<label>Split<select name="split" onchange="this.form.submit()"><option value="test"{" selected" if split == "test" else ""}>Test</option><option value="validation"{" selected" if split == "validation" else ""}>Validation</option></select></label>
<label>Metric<select name="metric">{_metric_options(split, metric)}</select></label><button type="submit">Update Chart</button></form>
<div class="chart-box">{chart}</div><h3>Current Run Metrics</h3>{all_metric_table}<h3>Recent Training History</h3>{history_table or '<p class="muted">No training history is available.</p>'}</section>'''




def _benchmark_integrity_warning(run_ids):
    """Return a visible warning when multiple model rows are numerically identical."""
    rows = []
    for run_id in run_ids:
        run = next((item for item in _completed_runs() if str(item.get("run_id")) == str(run_id)), None)
        if not run:
            continue
        values = tuple(_run_metric_value(str(run_id), key, "test") for key in ("dice", "iou", "precision", "recall", "specificity", "image_f1"))
        if all(value is not None for value in values):
            rows.append((str(run.get("model", "unknown")), values))
    if len(rows) < 2:
        return ""
    groups = {}
    for model, values in rows:
        groups.setdefault(values, []).append(model)
    repeated_groups = [models for models in groups.values() if len(models) >= 2]
    if repeated_groups:
        models = ", ".join(repeated_groups[0])
        return f'<div class="warning"><strong>Benchmark integrity warning:</strong> { _escape(models) } have identical core test metrics. Verify the checkpoints and training logs before using this chart as scientific evidence.</div>'
    return ""

def _render_compare(run_id, selected_metric="recall", selected_run_ids=None):
    current = _run_manifest(run_id) if run_id else None
    compatible = _compatible_runs(current or {}) if current else _completed_runs()
    compatible_ids = [str(run.get("run_id")) for run in compatible]
    selected = [rid for rid in (selected_run_ids or compatible_ids) if rid in compatible_ids]
    if not selected:
        selected = compatible_ids
    options = "".join(
        f'<label class="check"><input type="checkbox" name="run" value="{_escape(str(run.get("run_id")))}"{" checked" if str(run.get("run_id")) in selected else ""}> {_escape(str(run.get("model", "unknown")))} — {_escape(str(run.get("run_id")))}</label>'
        for run in compatible
    )
    values = []
    for run in compatible:
        rid = str(run.get("run_id"))
        if rid not in selected:
            continue
        value = _run_metric_value(rid, selected_metric, "test")
        if value is not None:
            values.append((f"{run.get('model', 'unknown')} {rid[-5:]}", value))
    chart = _svg_bar_chart(f"{_metric_label(selected_metric)} — selected runs", values, _metric_direction(selected_metric))
    core_chart = _svg_grouped_metric_chart("Core Model Comparison", selected, ["dice", "iou", "precision", "recall", "specificity", "image_f1"], "test")
    table = _metric_table(selected, ["dice", "iou", "positive_dice", "positive_iou", "hd95", "precision", "recall", "specificity", "mcc", "image_f1", "image_roc_auc", "image_pr_auc"], "test")
    warning = _benchmark_integrity_warning(selected)
    return f"""<section class=\"card\"><div class=\"section-heading\"><div><span class=\"eyebrow\">BENCHMARK</span><h2>Compare Models &amp; Runs</h2><p class=\"muted\">Completed runs using the same dataset contract can be compared side-by-side. Core scores are shown as grouped bars.</p></div></div>
<form method=\"get\" action=\"/compare\" class=\"compare-form\"><input type=\"hidden\" name=\"run_id\" value=\"{_escape(run_id or '')}\">
<label>Metric<select name=\"metric\">{"".join(f'<option value=\"{_escape(key)}\"{" selected" if key == selected_metric else ""}>{_escape(label)}</option>' for key,label in _METRICS)}</select></label>
<div class=\"run-checks\">{options or '<span class=\"muted\">No compatible completed runs yet.</span>'}</div><button type=\"submit\">Update Comparison</button></form>
{warning}<div class=\"chart-grid\"><div class=\"chart-box\">{core_chart}</div><div class=\"chart-box\">{chart}</div></div>
{table}<div class=\"note\"><strong>Direction:</strong> {_escape("higher is better" if _metric_direction(selected_metric) == "higher" else "lower is better")}. HD95, FPR and FNR are lower-is-better.</div></section>
{_connected_prediction_section("Model Outputs Linked to Benchmark Metrics", limit=8)}"""


def _render_diagnostics(run_id):
    manifest = _run_manifest(run_id) if run_id else None
    if not manifest:
        return '<section class="card"><h2>Diagnostics</h2><p class="muted">No run selected.</p></section>'
    path_map = _diagnostic_path_map(run_id)
    return render_diagnostics_html(
        run_id=run_id,
        manifest=manifest,
        metrics=_load_test_metrics(run_id),
        history=_load_training_history(run_id),
        paths=path_map,
        project_root=ROOT,
    )

def _safe_float(value):
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _project_file(path) -> Path | None:
    """Resolve a prediction artifact path while keeping links inside the project."""
    if not path:
        return None
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        candidate = candidate.resolve()
        candidate.relative_to(ROOT)
        return candidate if candidate.is_file() else None
    except (OSError, ValueError):
        # Ignore stale absolute paths from an older machine/installation.
        return None


def _prediction_records(limit=12):
    """Load generated prediction JSON files and attach model-level benchmark context."""
    prediction_root = ROOT / "predictions"
    if not prediction_root.is_dir():
        return []

    benchmark = {}
    for row in _load_csv_dicts(config.EXPERIMENTS_DIRECTORY / config.BENCHMARK_RESULTS_FILE):
        model = str(row.get("Model", "")).strip()
        run_id = str(row.get("Run ID", "")).strip()
        if not model:
            continue
        previous = benchmark.get(model)
        if previous is None or run_id > str(previous.get("Run ID", "")):
            benchmark[model] = row

    records = []
    json_files = sorted(ROOT.joinpath("predictions").glob("*_result.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    seen = set()
    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        model = str(payload.get("model", "unknown"))
        key = (str(payload.get("image", "")), model)
        if key in seen:
            continue
        seen.add(key)
        bench = benchmark.get(model, {})
        payload["_json_path"] = json_path
        payload["_overlay_path"] = _project_file(payload.get("overlay"))
        payload["_mask_path"] = _project_file(payload.get("mask"))
        payload["_report_path"] = _project_file(payload.get("report"))
        payload["_benchmark"] = {
            "run_id": str(bench.get("Run ID", "")),
            "dice": _safe_float(bench.get("Test Dice")),
            "iou": _safe_float(bench.get("Test IoU")),
            "precision": _safe_float(bench.get("Test Precision")),
            "recall": _safe_float(bench.get("Test Recall")),
            "specificity": _safe_float(bench.get("Test Specificity")),
            "image_f1": _safe_float(bench.get("Image F1")),
            "hd95": _safe_float(bench.get("Test HD95")),
        }
        records.append(payload)
        if len(records) >= limit:
            break
    return records


def _prediction_card(record, show_benchmark=True):
    model = str(record.get("model", "unknown"))
    prediction = str(record.get("prediction", "—"))
    probability = _safe_float(record.get("mean_pixel_probability"))
    area = _safe_float(record.get("predicted_area_percent"))
    inference = _safe_float(record.get("inference_time_ms"))
    overlay = record.get("_overlay_path")
    mask = record.get("_mask_path")
    report = record.get("_report_path")
    bench = record.get("_benchmark") or {}
    image_name = str(record.get("image", "Unknown image"))

    image_html = (
        f'<img src="/{quote(overlay.relative_to(ROOT).as_posix())}" alt="{_escape(model)} prediction overlay" loading="lazy">'
        if overlay
        else '<div class="empty">Overlay not generated.</div>'
    )
    links = []
    if mask:
        links.append(f'<a class="button-link secondary" href="/{quote(mask.relative_to(ROOT).as_posix())}" target="_blank">Mask</a>')
    if report:
        links.append(f'<a class="button-link secondary" href="/{quote(report.relative_to(ROOT).as_posix())}" target="_blank">PDF Report</a>')
    links.append(f'<a class="button-link secondary" href="/{quote(record["_json_path"].relative_to(ROOT).as_posix())}" target="_blank">JSON</a>')

    benchmark_html = ""
    if show_benchmark:
        def fmt(key):
            value = bench.get(key)
            return "—" if value is None else f"{value:.3f}"
        benchmark_html = f'''<div class="benchmark-context">
<div class="context-title">Benchmark context · {_escape(bench.get("run_id") or "not available")}</div>
<div class="mini-metrics">
<span>Dice <b>{fmt("dice")}</b></span>
<span>IoU <b>{fmt("iou")}</b></span>
<span>Precision <b>{fmt("precision")}</b></span>
<span>Recall <b>{fmt("recall")}</b></span>
<span>Specificity <b>{fmt("specificity")}</b></span>
<span>Image F1 <b>{fmt("image_f1")}</b></span>
</div>
<div class="note">Model-level test metrics shown for context; they are not per-image prediction scores.</div>
</div>'''

    badge = "positive" if prediction.lower() == "positive" else "negative"
    return f'''<article class="prediction-card">
<div class="prediction-image">{image_html}</div>
<div class="prediction-content">
<div class="prediction-head"><div><span class="eyebrow">AI OUTPUT</span><h3>{_escape(model.upper())}</h3><p class="muted">{_escape(image_name)}</p></div><span class="prediction-badge {badge}">{_escape(prediction)}</span></div>
<div class="output-kpis">
<div><span>Mean probability</span><b>{"—" if probability is None else f"{probability:.4f}"}</b></div>
<div><span>Predicted area</span><b>{"—" if area is None else f"{area:.3f}%"}</b></div>
<div><span>Inference</span><b>{"—" if inference is None else f"{inference:.1f} ms"}</b></div>
</div>
{benchmark_html}
<div class="card-actions">{" ".join(links)}</div>
</div></article>'''


def _connected_prediction_section(title="Latest Prediction Outputs", limit=8):
    records = _prediction_records(limit=limit)
    if not records:
        return f'''<section class="card"><div class="section-heading"><div><span class="eyebrow">OUTPUT → METRICS</span><h2>{_escape(title)}</h2><p class="muted">No prediction is stored yet. This is expected before you run inference on an X-ray.</p></div></div><div class="empty">No prediction is stored yet. This is expected before you run inference on an X-ray.</div></section>'''
    cards = "".join(_prediction_card(record) for record in records)
    return f'''<section class="card"><div class="section-heading"><div><span class="eyebrow">OUTPUT → METRICS</span><h2>{_escape(title)}</h2><p class="muted">Each image output is linked to its mask, PDF report and JSON result. Benchmark scores below are model-level reference metrics.</p></div><a class="button-link" href="/?view=prediction">Open Prediction</a></div><div class="prediction-grid">{cards}</div></section>'''

def _render_prediction(run_id=None):
    selected_model = config.MODEL
    runs = _prediction_runs()
    if run_id:
        selected_model = str((_run_manifest(run_id) or {}).get("model", selected_model))
    options = '<option value="">Auto / latest checkpoint</option>' + "".join(
        f'<option value="{_escape(str(run.get("run_id")))}"{" selected" if str(run.get("run_id")) == run_id else ""}>{_escape(str(run.get("model", "unknown")))} — {_escape(str(run.get("run_id")))}</option>'
        for run in runs
    )
    default_output = str((ROOT / "predictions").resolve())
    return f'''<section class="card"><div class="section-heading"><div><span class="eyebrow">INFERENCE</span><h2>Generate X-ray Prediction Report</h2><p class="muted">Upload an X-ray directly from this dashboard. Drag &amp; drop it into the upload box or click <strong>Browse X-ray</strong>.</p></div><span class="status-pill">Research only</span></div>
<form method="post" action="/predict" enctype="multipart/form-data" id="prediction-form">
<div class="upload-zone" id="upload-zone" tabindex="0" role="button" aria-label="Upload an X-ray image">
<input id="xray-file" name="xray_file" type="file" accept=".dcm,.dicom,.png,.jpg,.jpeg,.bmp" required>
<div class="upload-icon">+</div><div class="upload-title">Drop your X-ray here</div><div class="upload-subtitle">or click to browse • DICOM, PNG, JPG, JPEG, BMP • up to 50 MB</div>
<div class="upload-file" id="upload-file">No file selected</div>
</div>
<div class="form-grid prediction-controls">
<label>Completed run<select name="run_id">{options}</select></label>
<label>Model<select name="model">{_options(_MODELS, selected_model)}</select></label>
{_field("Segmentation threshold", "threshold", 0.50, "number", 0, "any")}
<label>Device<select name="device">{_options(_DEVICES, "auto")}</select></label>
{_field("Output directory", "output_dir", default_output)}
</div><div class="checks-row">{_checkbox("all_models", "Run all available model checkpoints")}</div><button type="submit" id="predict-button">Generate Prediction Report</button></form>
<p class="note">The browser uploads the selected X-ray to the local inference service, then runs the selected trained checkpoint(s). The generated mask, boundary overlay, quality analysis, metrics and PDF report appear in the gallery below.</p>
</section>
{_connected_prediction_section("Prediction Output Gallery", limit=8)}
<script>
(() => {{
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('xray-file');
  const label = document.getElementById('upload-file');
  const form = document.getElementById('prediction-form');
  if (!zone || !input) return;
  const update = () => {{
    const file = input.files && input.files[0];
    label.textContent = file ? `${{file.name}} • ${{Math.max(0.1, file.size / 1048576).toFixed(1)}} MB` : 'No file selected';
    zone.classList.toggle('has-file', !!file);
  }};
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (event) => {{ if (event.key === 'Enter' || event.key === ' ') {{ event.preventDefault(); input.click(); }} }});
  ['dragenter','dragover'].forEach(name => zone.addEventListener(name, (event) => {{ event.preventDefault(); zone.classList.add('dragover'); }}));
  ['dragleave','drop'].forEach(name => zone.addEventListener(name, (event) => {{ event.preventDefault(); zone.classList.remove('dragover'); }}));
  zone.addEventListener('drop', (event) => {{
    const files = event.dataTransfer.files;
    if (!files || !files.length) return;
    try {{ input.files = files; }} catch (_) {{ return; }}
    update();
  }});
  input.addEventListener('change', update);
  form.addEventListener('submit', () => {{
    const button = document.getElementById('predict-button');
    if (button) {{ button.disabled = true; button.textContent = 'Running inference…'; }}
  }});
}})();
</script>'''

def render_index(message="", selected_run_id=None, view="dashboard", split="test", metric="recall", compare_metric="recall", compare_runs=None) -> bytes:
    active_status, active_exit, active_run_id = _training_status()
    run_id = selected_run_id or active_run_id or (_list_runs()[0]["run_id"] if _list_runs() else None)
    manifest = _run_manifest(run_id) if run_id else None
    paths = _get_run_paths(run_id) if run_id else None
    status = manifest.get("status") if manifest else "NO RUN"
    message_html = f'<div class="message">{_escape(message)}</div>' if message else ""
    selected_model = (manifest or {}).get("model", config.MODEL)
    process_state = _read_process_state(run_id) if run_id else {}
    if active_run_id and run_id == active_run_id:
        epoch_text = ""
        if process_state.get("epoch") is not None:
            epoch_text = f'<p><strong>Epoch:</strong> {_escape(process_state.get("epoch"))}</p>'
        body = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<title>MedPlugBench — Training</title><meta http-equiv="refresh" content="3">'
            f'<style>{_css()}</style></head><body><header><h1>MedPlugBench</h1>{_nav("dashboard", run_id)}</header>'
            f'<section class="card hero"><div><h2>Training is running in PowerShell</h2>'
            f'<p><strong>Run:</strong> {_escape(run_id)}</p>{epoch_text}'
            '<p class="muted">Live training output is being forwarded to this PowerShell window and persisted in the run log.</p>'
            f'</div></section><section class="card"><h3>Process State</h3><pre>{_escape(json.dumps(process_state or {"status":"RUNNING"}, indent=2))}</pre></section>'
            f'<section class="card"><h3>Live Training Log</h3><pre>{_escape(_log_tail(run_id, 80))}</pre></section></body></html>'
        )

        return body.encode("utf-8")

    if view == "setup":
        content = _setup_section(selected_model)
    elif view == "analysis":
        content = (_metric_audit_note(run_id) + _render_analysis_form(run_id, split, metric)) if run_id else '<section class="card"><h2>Metric Explorer</h2><p class="muted">No completed run selected.</p></section>'
    elif view == "compare":
        content = _render_compare(run_id, compare_metric, compare_runs) if run_id else '<section class="card"><h2>Compare Models</h2><p class="muted">No completed run selected.</p></section>'
    elif view == "diagnostics":
        content = _render_diagnostics(run_id) if run_id else '<section class="card"><h2>Diagnostics</h2><p class="muted">No run selected.</p></section>'
    elif view == "research":
        content = '<section class="card"><span class="eyebrow">MANUSCRIPT OUTPUTS</span><h2>Research Figures</h2><p class="muted">Figures are rebuilt from persisted per-image metrics and epoch diagnostics. The dashboard keeps protocol caveats and missing evidence visible.</p></section>' + render_research_figures(ROOT)
    elif view == "prediction":
        content = _render_prediction(run_id)
    else:
        metrics_table = _metric_table([run_id], list(dict(_METRICS).keys()), "test") if run_id else '<p class="muted">No metrics available.</p>'
        visuals = "" if not paths else "".join([
            _image_card(run_id, paths.overlay_path, "Prediction Overlay"),
            _image_card(run_id, paths.prediction_path, "Prediction Mask"),
            _image_card(run_id, paths.ground_truth_path, "Ground Truth Mask"),
            _image_card(run_id, paths.training_loss_plot_path, "Training Loss"),
            _image_card(run_id, paths.validation_metrics_plot_path, "Validation Metrics"),
            _image_card(run_id, paths.test_metrics_plot_path, "Final Test Metrics"),
            _image_card(run_id, paths.model_comparison_path, "Model Comparison"),
        ])
        report_path = paths.root / config.REPORT_FILE if paths else None
        report_text = report_path.read_text(encoding="utf-8", errors="replace") if report_path and report_path.exists() else "No training report is available for this run yet. A report is created after the independent pipeline completes evaluation, visualization and report generation."
        prepared_command = ""
        command_path = paths.root / "logs" / "prepared_command.txt" if paths else None
        if command_path and command_path.is_file():
            prepared_command = command_path.read_text(encoding="utf-8", errors="replace").strip()
        command_section = ""
        if status == "READY" and prepared_command:
            command_section = f"""<section class=\"card\"><h2>PowerShell Command</h2><p class=\"muted\">Preflight passed. Close app.py, open PowerShell, and paste this exact command.</p><pre>{_escape(prepared_command)}</pre><button type=\"button\" onclick=\"navigator.clipboard.writeText({json.dumps(prepared_command)})\">Copy Command</button></section>"""
        content = f'''<section class="hero card"><div><h2>{_escape((manifest or {}).get("model", "No run selected")).upper()}</h2><p class="muted">Run {_escape(run_id or "none")} · Status <span class="status-pill">{_escape(status)}</span></p></div><div class="kpi-grid">{_kpi_cards(run_id) if run_id else ""}</div></section>
<section class="card"><h2>Run History</h2>{_runs_table(run_id)}</section>
{_metric_audit_note(run_id) if run_id else ""}
{command_section}
<section class="card"><h2>Evaluation Metrics</h2>{metrics_table}<p><a class="button-link" href="/?view=analysis&amp;run_id={quote(run_id or "")}">Explore Metrics</a> <a class="button-link" href="/?view=compare&amp;run_id={quote(run_id or "")}">Compare Models</a> <a class="button-link" href="/?view=research&amp;run_id={quote(run_id or "")}">Research Figures</a> <a class="button-link" href="/?view=prediction&amp;run_id={quote(run_id or "")}">Generate Prediction</a> <a class="button-link" href="/?view=diagnostics&amp;run_id={quote(run_id or "")}">Diagnostics</a></p></section>
<section class="card"><h2>Current Run Visual Outputs</h2><div class="grid">{visuals}</div><p class="note">These are the artifacts generated by the selected training run.</p></section>
{render_research_figures(ROOT)}
{_connected_prediction_section("Connected AI Outputs", limit=4)}
<section class="card"><h2>Artifacts</h2>{_artifact_browser(paths) if paths else '<p class="muted">No run selected.</p>'}</section>
<section class="card"><h2>Latest Report</h2><pre>{_escape(report_text)}</pre></section>'''

    workflow = '<div class="workflow"><span class="workflow-label">WORKFLOW</span><span class="workflow-step done">Train</span><span class="workflow-arrow">→</span><span class="workflow-step done">Evaluate</span><span class="workflow-arrow">→</span><span class="workflow-step done">Visualize</span><span class="workflow-arrow">→</span><span class="workflow-step active">Predict</span><span class="workflow-arrow">→</span><span class="workflow-step active">Compare</span></div>'
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>MedPlugBench</title><style>{_css()}</style></head><body><header><h1>MedPlugBench</h1><p class="muted">Research experiment control, analysis and comparison</p>{_nav(view, run_id)}{workflow}</header>{message_html}{content}</body></html>'''
    return body.encode("utf-8")


def _css():
    return """:root{font-family:Inter,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#0f172a;background:#f1f5f9;line-height:1.45}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#f8fafc 0%,#eef2f7 100%);color:#0f172a}
header{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);backdrop-filter:blur(10px);border-bottom:1px solid #dbe3ec;padding:16px 28px;box-shadow:0 2px 16px rgba(15,23,42,.05)}
header h1{margin:0;font-size:21px;letter-spacing:-.02em} header .muted{margin:3px 0 0;font-size:13px}
nav{display:flex;gap:7px;flex-wrap:wrap;margin-top:13px}
.nav-link,.button-link{display:inline-flex;align-items:center;justify-content:center;min-height:36px;padding:8px 13px;border:1px solid #d6dee8;border-radius:9px;text-decoration:none;background:#fff;color:#334155;font-size:13px;font-weight:600;transition:.15s}
.nav-link:hover,.button-link:hover{border-color:#94a3b8;background:#f8fafc}.nav-link.active{background:#0f172a;color:#fff;border-color:#0f172a}
.card{width:min(1240px,calc(100% - 44px));margin:18px auto;padding:22px;border:1px solid #d9e1ea;border-radius:14px;background:#fff;box-shadow:0 5px 24px rgba(15,23,42,.055)}
.hero{display:flex;justify-content:space-between;gap:22px;align-items:flex-start}.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
.eyebrow{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.16em;color:#64748b;margin-bottom:5px}.muted{color:#64748b}.note{font-size:12px;color:#64748b;margin-top:11px}
.status-pill{display:inline-flex;align-items:center;background:#ecfdf5;color:#166534;border:1px solid #bbf7d0;border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(125px,1fr));gap:10px;min-width:570px}.kpi{background:#f8fafc;border:1px solid #dbe3ec;border-radius:10px;padding:12px}.kpi-label{font-size:11px;color:#64748b}.kpi-value{font-size:22px;font-weight:750;margin-top:3px;letter-spacing:-.02em}
.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:18px}.chart-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px;margin-top:16px}.grid img{width:100%;height:auto;display:block;border:1px solid #dbe3ec;border-radius:9px;background:#f8fafc}
.form-grid{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:14px}.inline-form,.explore-form,.compare-form{display:flex;gap:12px;align-items:end;flex-wrap:wrap}
.inline-form label,.explore-form label,.compare-form label,.form-grid label{display:block;font-size:13px;font-weight:650;color:#334155}.form-grid input,.form-grid select,.inline-form input,.inline-form select,.explore-form select,.compare-form select{width:100%;min-height:38px;margin-top:6px;padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;color:#0f172a;outline:none}
.form-grid input:focus,.form-grid select:focus,.inline-form input:focus,.inline-form select:focus,.explore-form select:focus,.compare-form select:focus{border-color:#64748b;box-shadow:0 0 0 3px rgba(100,116,139,.12)}.inline-form label,.explore-form label,.compare-form label{min-width:220px}
.checks-row{display:flex;gap:16px;flex-wrap:wrap;margin:15px 0}.check{font-weight:500;margin:4px 0}.check input{width:auto;margin-right:7px;accent-color:#0f172a}.metric-actions{display:flex;gap:8px;margin:8px 0 10px}.metric-actions button{margin-top:0;background:#475569}.metric-grid{columns:3 230px}.metric-grid .check{break-inside:avoid}
.run-checks{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:7px;width:100%;padding:11px;background:#f8fafc;border:1px solid #dbe3ec;border-radius:10px}.chart-box{margin-top:16px;border:1px solid #dbe3ec;border-radius:11px;padding:10px;background:#fff;overflow:auto}.chart-box svg{width:100%;height:auto;display:block}
.table-scroll{width:100%;max-width:100%;overflow-x:auto;overflow-y:hidden;margin-top:12px;border:1px solid #dbe3ec;border-radius:10px;-webkit-overflow-scrolling:touch}.data-table{width:max-content;min-width:100%;border-collapse:collapse;margin:0}.data-table th,.data-table td{border-bottom:1px solid #e5e7eb;padding:10px 11px;text-align:left;vertical-align:top;white-space:nowrap;font-size:12px}.data-table th{background:#f8fafc;color:#334155;font-weight:750;position:sticky;top:0;z-index:1}.data-table tr:last-child td{border-bottom:0}.data-table th:first-child,.data-table td:first-child{position:sticky;left:0;background:#fff;z-index:2}.data-table th:first-child{background:#f8fafc;z-index:3}
.diag-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 18px}.diag-tabs a{padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;text-decoration:none;background:#f8fafc;color:#334155}.diag-tabs a:hover{background:#e2e8f0}
.message{width:min(1240px,calc(100% - 44px));margin:18px auto;padding:12px 15px;border:1px solid #bbf7d0;background:#f0fdf4;color:#166534;border-radius:10px}.warning{margin:14px 0;padding:12px 14px;border:1px solid #f3d18a;background:#fff8e1;color:#854d0e;border-radius:10px;font-size:12px}.empty{padding:16px;color:#64748b;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px}
button,.button-link{border:1px solid #0f172a;background:#0f172a;color:#fff;padding:9px 15px;border-radius:8px;cursor:pointer;font-weight:650}button:hover,.button-link:hover{background:#1e293b;color:#fff}a{color:#1d4ed8}pre{white-space:pre-wrap;line-height:1.45;max-height:700px;overflow:auto;background:#0b1220;color:#e2e8f0;padding:14px;border-radius:10px;border:1px solid #1e293b}
h2{font-size:19px;letter-spacing:-.02em;margin:0 0 8px}h3{font-size:14px;margin:18px 0 8px}
.workflow{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-top:12px;padding-top:11px;border-top:1px solid #e5e7eb}.workflow-label{font-size:10px;font-weight:800;letter-spacing:.12em;color:#64748b;margin-right:2px}.workflow-step{font-size:11px;font-weight:700;padding:5px 9px;border-radius:999px;border:1px solid #dbe3ec;background:#f8fafc;color:#64748b}.workflow-step.done{color:#334155}.workflow-step.active{background:#0f172a;color:#fff;border-color:#0f172a}.workflow-arrow{color:#94a3b8}
.prediction-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:14px}.prediction-card{display:grid;grid-template-columns:minmax(190px,.95fr) minmax(250px,1.05fr);gap:15px;border:1px solid #dbe3ec;border-radius:12px;padding:12px;background:#fff;overflow:hidden}.prediction-image img{width:100%;height:100%;min-height:190px;object-fit:cover;border:1px solid #dbe3ec;border-radius:9px;background:#0b1220}.prediction-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.prediction-head h3{margin:0;font-size:15px}.prediction-badge{display:inline-flex;padding:5px 9px;border-radius:999px;border:1px solid #cbd5e1;font-size:10px;font-weight:800}.prediction-badge.positive{background:#ecfdf5;color:#166534;border-color:#bbf7d0}.prediction-badge.negative{background:#f8fafc;color:#475569}.output-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}.output-kpis div{padding:9px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px}.output-kpis span{display:block;font-size:10px;color:#64748b}.output-kpis b{display:block;margin-top:2px;font-size:15px}.benchmark-context{border-top:1px solid #e5e7eb;padding-top:10px}.context-title{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#64748b;font-weight:800;margin-bottom:8px}.mini-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.mini-metrics span{font-size:10px;color:#64748b}.mini-metrics b{color:#0f172a}.card-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.button-link.secondary{background:#fff;color:#334155;border-color:#cbd5e1}.drop-hint{margin:14px 0;padding:12px 14px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc;color:#334155;font-size:12px}.drop-hint code{font-family:Consolas,monospace;background:#e2e8f0;padding:2px 5px;border-radius:4px}.upload-zone{position:relative;margin:4px 0 18px;padding:34px 22px;text-align:center;border:2px dashed #cbd5e1;border-radius:14px;background:linear-gradient(180deg,#fbfdff,#f8fafc);transition:.15s ease;cursor:pointer}.upload-zone:hover,.upload-zone.dragover{border-color:#475569;background:#f1f5f9;box-shadow:0 0 0 4px rgba(71,85,105,.08)}.upload-zone.has-file{border-style:solid;border-color:#86efac;background:#f0fdf4}.upload-zone input[type=file]{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}.upload-icon{width:38px;height:38px;display:inline-flex;align-items:center;justify-content:center;border:1px solid #cbd5e1;border-radius:10px;background:#fff;font-size:25px;color:#334155;margin-bottom:8px}.upload-title{font-size:16px;font-weight:800;color:#0f172a}.upload-subtitle{margin-top:5px;font-size:12px;color:#64748b}.upload-file{margin:12px auto 0;width:min(560px,100%);padding:8px 10px;border-radius:8px;background:#fff;border:1px solid #e2e8f0;color:#475569;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.upload-zone.has-file .upload-file{border-color:#bbf7d0;color:#166534}.prediction-controls{grid-template-columns:repeat(4,minmax(180px,1fr))}.prediction-controls label:last-child{grid-column:span 2}#predict-button:disabled{opacity:.65;cursor:wait}
@media(max-width:1000px){.hero,.section-heading{display:block}.kpi-grid{min-width:0;margin-top:18px;grid-template-columns:repeat(2,minmax(120px,1fr))}.form-grid{grid-template-columns:repeat(2,minmax(210px,1fr))}.chart-grid{grid-template-columns:1fr}.prediction-grid{grid-template-columns:1fr}}
@media(max-width:650px){header{padding:14px 15px}.card,.message{width:calc(100% - 24px);margin:12px auto;padding:17px}.grid,.form-grid{grid-template-columns:1fr}.kpi-grid{grid-template-columns:1fr 1fr}.prediction-card{grid-template-columns:1fr}.output-kpis,.mini-metrics{grid-template-columns:1fr 1fr}.inline-form label,.explore-form label,.compare-form label{min-width:100%}}
"""


def _parse_prediction_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], tuple[str, bytes] | None]:
    if len(body) > _MAX_UPLOAD_BYTES:
        raise ValueError(f"Upload is too large. Maximum size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=email_default_policy).parsebytes(header + body)
    fields: dict[str, str] = {}
    upload: tuple[str, bytes] | None = None
    for part in message.walk():
        if part.is_multipart():
            continue
        disposition = part.get_content_disposition()
        if disposition != "form-data":
            continue
        name = part.get_param("name", header="Content-Disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None and name == "xray_file":
            upload = (Path(filename).name, payload)
        elif name:
            charset = part.get_content_charset() or "utf-8"
            fields[str(name)] = payload.decode(charset, errors="replace")
    return fields, upload


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        query = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
        if path in {"/", "/compare"}:
            view = "compare" if path == "/compare" else query.get("view", ["dashboard"])[0]
            payload = render_index(
                message=query.get("message", [""])[0],
                selected_run_id=query.get("run_id", [None])[0],
                view=view,
                split=query.get("split", ["test"])[0],
                metric=query.get("metric", ["recall"])[0],
                compare_metric=query.get("metric", ["recall"])[0],
                compare_runs=query.get("run", []),
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        relative = path.lstrip("/")
        file_path = (ROOT / relative).resolve()
        try:
            file_path.relative_to(ROOT)
        except ValueError:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        suffix = file_path.suffix.lower()
        content_type = {".png": "image/png", ".csv": "text/csv; charset=utf-8", ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8", ".yaml": "text/plain; charset=utf-8", ".json": "application/json", ".pdf": "application/pdf"}.get(suffix, "application/octet-stream")
        payload = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        target = self.path.split("?", 1)[0]
        if target == "/predict":
            try:
                content_type = self.headers.get("Content-Type", "")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Please select an X-ray before generating a report.")
                if length > _MAX_UPLOAD_BYTES + 1024 * 1024:
                    raise ValueError(f"Upload is too large. Maximum size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                body = self.rfile.read(length)
                if content_type.lower().startswith("multipart/form-data"):
                    form, upload = _parse_prediction_multipart(body, content_type)
                else:
                    form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
                    form = {key: values[0] if values else "" for key, values in form.items()}
                    upload = None

                if upload is None or not upload[0]:
                    raise ValueError("Please choose or drag an X-ray into the upload area.")
                original_name, image_bytes = upload
                suffix = Path(original_name).suffix.lower()
                if suffix not in _ALLOWED_IMAGE_SUFFIXES:
                    raise ValueError("Unsupported file type. Use DICOM, PNG, JPG, JPEG, or BMP.")
                if not image_bytes:
                    raise ValueError("The uploaded file is empty.")

                upload_dir = ROOT / "predictions" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(original_name).stem).strip("_") or "xray"
                upload_path = upload_dir / f"{timestamp}_{safe_stem}{suffix}"
                upload_path.write_bytes(image_bytes)

                run_id = str(form.get("run_id", "")).strip()
                model = str(form.get("model", config.MODEL)).strip() or config.MODEL
                output_dir = str(form.get("output_dir", str(ROOT / "predictions"))).strip() or str(ROOT / "predictions")
                threshold = str(form.get("threshold", str(config.SEGMENTATION_THRESHOLD)))
                device = str(form.get("device", "auto"))
                all_models = bool(form.get("all_models"))

                command = [sys.executable, str(ROOT / "predict.py"), "--image", str(upload_path), "--output-dir", output_dir, "--device", device, "--threshold", threshold]
                if all_models:
                    command.append("--all-models")
                else:
                    command.extend(["--model", model])
                    if run_id:
                        command.extend(["--run-id", run_id])
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
                output = ((result.stdout or "") + (result.stderr or "")).strip()
                message = "Prediction report generated successfully." if result.returncode == 0 else "Prediction failed."
                if output:
                    message += " " + output.splitlines()[-1][:400]
            except Exception as exc:
                message = f"Prediction error: {type(exc).__name__}: {exc}"
            query = urlencode({"message": message, "run_id": run_id if 'run_id' in locals() else "", "view": "prediction"})
            self.send_response(303)
            self.send_header("Location", "/?" + query)
            self.end_headers()
            return

        if target != "/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body, keep_blank_values=True)

        class Form:
            def getfirst(self, key, default=""):
                values = form.get(key)
                return values[0] if values else default
            def __contains__(self, key):
                return key in form

        wrapped = Form()
        paths = None
        try:
            dataset_root = Path(_normalize_dataset_root(wrapped.getfirst("dataset_root", str(_default_dataset_root()))))
            if not str(dataset_root):
                raise FileNotFoundError(
                    "No dataset root was configured. Set the Dataset root or dot-source tools\\set_dataset_root.ps1 before starting app.py."
                )
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
            if not _has_dataset_layout(dataset_root):
                raise FileNotFoundError(
                    "The selected dataset root is incomplete. Expected data\\train-rle.csv, "
                    "data\\val-rle.csv, data\\test-rle.csv and either "
                    "images\\dicom-images-train or aligned_dataset\\images + aligned_dataset\\masks."
                )
            original_getfirst = wrapped.getfirst
            wrapped.getfirst = lambda key, default="": str(dataset_root) if key == "dataset_root" else original_getfirst(key, default)
            selected_model = wrapped.getfirst("model", config.MODEL)
            paths = _MANAGER.create_experiment(
                selected_model,
                checkpoint_name=f"{selected_model}_best_model",
                resume=False,
            )
            config_path = _write_ui_config(wrapped, paths)
            passed, preflight_message, preflight_output = _run_preflight(config_path)
            (paths.root / "logs" / "preflight.log").write_text(preflight_output, encoding="utf-8")
            if not passed:
                _MANAGER.update_stage(paths, "PREFLIGHT", status="FAILED", error=preflight_message)
                _write_process_state(paths, "PREFLIGHT_FAILED", stage="PREFLIGHT", error=preflight_message)
                message = f"{paths.run_id}: {preflight_message}"
            else:
                _MANAGER.update_stage(paths, "READY", status="READY")
                command = _prepare_run(paths, config_path)
                (paths.root / "logs" / "prepared_command.txt").write_text(command + "\n", encoding="utf-8")
                message = f"{paths.run_id}: READY. Close app.py and run the saved PowerShell command."
        except Exception as exc:
            message = f"Could not prepare run: {type(exc).__name__}: {exc}"
            if paths is not None:
                _MANAGER.mark_failed(paths, message)
                _write_process_state(paths, "FAILED", stage="PREPARATION", error=message)
        query = urlencode({"message": message, "run_id": paths.run_id if paths else "", "view": "dashboard"})
        self.send_response(303)
        self.send_header("Location", "/?" + query)
        self.end_headers()

    def log_message(self, fmt, *args):
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"MedPlugBench control panel: {url}")
    print("Each new experiment is isolated under experiments/RUN_.../.")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
