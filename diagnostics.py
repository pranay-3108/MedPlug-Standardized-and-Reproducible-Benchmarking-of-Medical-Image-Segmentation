"""Research-oriented diagnostics built only from persisted experiment artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path


METRIC_LABELS = {
    "Test Dice (All)": "Dice (all)",
    "Test Dice (Positive)": "Dice (positive)",
    "Test IoU (All)": "IoU (all)",
    "Test IoU (Positive)": "IoU (positive)",
    "Test HD95": "HD95",
    "Test Precision": "Precision",
    "Test Recall": "Recall",
    "Test Specificity": "Specificity",
    "Test MCC": "MCC",
    "Test Pixel Accuracy": "Pixel Accuracy",
    "Test FPR": "FPR",
    "Test FNR": "FNR",
    "Image F1": "Image F1",
    "Image Accuracy": "Image Accuracy",
    "Image Precision": "Image Precision",
    "Image Recall": "Image Recall",
    "Image Specificity": "Image Specificity",
    "Image ROC-AUC": "Image ROC-AUC",
    "Image PR-AUC": "Image PR-AUC",
}

MODEL_FILES = {
    "unet": "models/unet.py",
    "attention_unet": "models/attention_unet.py",
    "transunet": "models/transunet.py",
    "segformer": "models/segformer.py",
}


def load_metrics(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.is_file():
        return result
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("Metric", "")).strip()
                if not name or name == "Selected Metrics":
                    continue
                try:
                    result[name] = float(row.get("Value", ""))
                except (TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return {}
    return result


def load_history(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def _floats(history: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in history:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def training_stability(history: list[dict[str, str]]) -> dict:
    if not history:
        return {"available": False}

    losses = _floats(history, "TrainLoss")
    dice = _floats(history, "ValidationDicePositive")
    iou = _floats(history, "ValidationIoUPositive")
    learning_rates = _floats(history, "LearningRate")

    best_epoch = None
    best_dice = None
    if dice:
        best_index = max(range(len(dice)), key=lambda index: dice[index])
        best_dice = dice[best_index]
        try:
            best_epoch = int(history[best_index].get("Epoch", best_index + 1))
        except (TypeError, ValueError):
            best_epoch = best_index + 1

    loss_deltas = [losses[index] - losses[index - 1] for index in range(1, len(losses))]
    dice_deltas = [dice[index] - dice[index - 1] for index in range(1, len(dice))]
    oscillations = sum(
        1 for index in range(1, len(dice_deltas))
        if dice_deltas[index] * dice_deltas[index - 1] < 0
    )

    first_loss = losses[0] if losses else None
    last_loss = losses[-1] if losses else None
    loss_drop = (first_loss - last_loss) if first_loss is not None and last_loss is not None else None
    dice_std = _std(dice)
    iou_std = _std(iou)

    return {
        "available": True,
        "epochs": len(history),
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "latest_loss": last_loss,
        "first_loss": first_loss,
        "loss_drop": loss_drop,
        "dice_mean": _mean(dice),
        "dice_std": dice_std,
        "dice_min": min(dice) if dice else None,
        "dice_max": max(dice) if dice else None,
        "iou_std": iou_std,
        "loss_delta_std": _std(loss_deltas),
        "dice_delta_std": _std(dice_deltas),
        "oscillations": oscillations,
        "learning_rate_start": learning_rates[0] if learning_rates else None,
        "learning_rate_end": learning_rates[-1] if learning_rates else None,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5



def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import json
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}


def load_case_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def exact_error_profile(records: list[dict]) -> dict:
    if not records:
        return {"available": False}
    counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for record in records:
        for key in counts:
            try:
                counts[key] += int(float(record.get(key.upper(), 0) or 0))
            except (TypeError, ValueError):
                pass
    return {
        "available": True,
        "records": len(records),
        **counts,
        "note": "Computed directly from persisted per-image test prediction records.",
    }

def metric_warnings(metrics: dict[str, float]) -> list[str]:
    warnings = []
    positive_dice = metrics.get("Test Dice (Positive)")
    positive_iou = metrics.get("Test IoU (Positive)")
    recall = metrics.get("Image Recall")
    precision = metrics.get("Image Precision")
    specificity = metrics.get("Image Specificity")
    if positive_dice is not None and positive_dice < 0.10:
        warnings.append("Positive-region Dice is very low; inspect lesion localization and prediction masks.")
    if positive_iou is not None and positive_iou < 0.05:
        warnings.append("Positive-region IoU is very low; overlap with ground truth is poor.")
    if recall is not None and precision is not None and recall > 0.90 and precision < 0.30:
        warnings.append("High image-level recall with low precision indicates many false-positive images.")
    if specificity is not None and specificity < 0.20:
        warnings.append("Image-level specificity is very low; the model is weak on negative cases.")
    if not warnings:
        warnings.append("No threshold-based warning was triggered by the persisted test metrics.")
    return warnings


def derived_error_profile(metrics: dict[str, float], positive_count: int = 174, negative_count: int = 680) -> dict:
    """Derive an approximate image-level confusion profile from persisted recall/specificity."""
    recall = metrics.get("Image Recall")
    specificity = metrics.get("Image Specificity")
    if recall is None or specificity is None:
        return {"available": False}

    tp = recall * positive_count
    fn = positive_count - tp
    tn = specificity * negative_count
    fp = negative_count - tn
    return {
        "available": True,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "note": "Derived from image-level recall/specificity and the validated test split counts; values are approximate due to metric rounding.",
    }


def artifact_health(paths: dict[str, Path], manifest_artifacts: dict[str, str] | None = None, run_root: Path | None = None) -> dict:
    """Check each expected artifact on disk, and report enough detail to
    debug a mismatch instead of just a bare count.

    manifest_artifacts/run_root are optional: when given, each missing
    artifact also reports what the run's manifest.json last recorded for
    that key (relative to run_root), so a path-resolution drift between
    "what the app looks for now" and "what training actually wrote" is
    visible directly in the diagnostics page rather than only in logs.
    """
    manifest_artifacts = manifest_artifacts or {}
    missing_detail = []
    existing = 0
    for key, path in paths.items():
        if path.is_file():
            existing += 1
            continue
        recorded = manifest_artifacts.get(key)
        recorded_path = None
        recorded_exists = None
        if recorded and run_root is not None:
            # Manifest-recorded artifact paths may use '/' or '\\' depending
            # on the OS that produced them; normalize before resolving.
            recorded_path = run_root / recorded.replace("\\", "/")
            recorded_exists = recorded_path.is_file()
        missing_detail.append({
            "key": key,
            "checked_path": str(path),
            "manifest_recorded": recorded,
            "manifest_path_exists": recorded_exists,
        })
    total = len(paths)
    return {
        "existing": existing,
        "total": total,
        "missing": total - existing,
        "missing_detail": missing_detail,
    }


def architecture_audit(project_root: Path, model: str) -> dict:
    relative = MODEL_FILES.get(model)
    if not relative:
        return {"status": "UNKNOWN", "reason": "No source mapping for this model."}
    source_path = project_root / relative
    if not source_path.is_file():
        return {"status": "MISSING", "path": str(source_path)}
    try:
        source = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "UNREADABLE", "path": str(source_path), "reason": str(exc)}

    placeholder_markers = ("placeholder", "simple projection", "temporary implementation", "not a full")
    lowered = source.lower()
    hits = [marker for marker in placeholder_markers if marker in lowered]
    conv_count = source.count("nn.Conv2d")
    has_forward = "def forward" in source
    has_attention = "attention" in lowered or "multihead" in lowered
    has_transformer = "transformer" in lowered or "self-attention" in lowered or "attention" in lowered

    status = "REVIEW"
    if hits:
        status = "PLACEHOLDER / REVIEW"
    elif model == "unet" and has_forward and conv_count >= 5:
        status = "IMPLEMENTED (source-level heuristic)"

    return {
        "status": status,
        "path": str(source_path),
        "placeholder_hits": ", ".join(hits) or "none",
        "conv2d_count": conv_count,
        "has_forward": has_forward,
        "has_attention_like_code": has_attention,
        "has_transformer_like_code": has_transformer,
        "note": "Source-level audit only; it does not prove architectural correctness.",
    }


def explainability_readiness(project_root: Path, model: str, run_paths: dict[str, Path]) -> dict:
    audit = architecture_audit(project_root, model)
    existing_visuals = [
        run_paths.get("overlay"),
        run_paths.get("prediction"),
        run_paths.get("ground_truth"),
    ]
    has_visuals = all(path and path.is_file() for path in existing_visuals)
    return {
        "model": model,
        "base_visuals_ready": has_visuals,
        "gradcam_available": False,
        "layer_hooks_available": False,
        "attention_source_signal": bool(audit.get("has_attention_like_code")),
        "note": "Grad-CAM/layer attribution is not generated automatically yet. This panel reports readiness without inventing explanations.",
    }


def _fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_diagnostics_html(*, run_id: str, manifest: dict, metrics: dict[str, float], history: list[dict[str, str]], paths: dict[str, Path], project_root: Path) -> str:
    model = str(manifest.get("model", "unknown"))
    status = str(manifest.get("status", "UNKNOWN"))
    stability = training_stability(history)
    run_root = paths.get("_run_root")
    case_records = load_case_records(paths.get("test_predictions", Path(""))) if run_root else []
    errors = exact_error_profile(case_records) if case_records else derived_error_profile(metrics)
    audit = architecture_audit(project_root, model)
    explain = explainability_readiness(project_root, model, paths)
    warnings = metric_warnings(metrics)
    verification = load_json(paths.get("verification", Path("")))
    signature = load_json(paths.get("model_signature", Path("")))
    trace = load_json(paths.get("run_forensics", Path("")))
    artifact_paths = {key: value for key, value in paths.items() if key != "_run_root"}
    health = artifact_health(artifact_paths, manifest.get("artifacts"), run_root)

    warning_rows = "".join(f"<li>{_esc(item)}</li>" for item in warnings)
    if not metrics:
        warning_rows += (
            "<li><strong>metrics.csv could not be loaded</strong> — the checks above ran "
            "against an empty metric set, so \"no warning triggered\" does not mean the "
            "run's metrics are healthy. See Artifact Health for the exact path checked.</li>"
        )
    missing_detail_rows = "".join(
        f"<li><code>{_esc(item['key'])}</code>: checked <code>{_esc(item['checked_path'])}</code>"
        + (
            f" — manifest recorded <code>{_esc(item['manifest_recorded'])}</code>"
            f" ({'found there' if item['manifest_path_exists'] else 'also missing there'})"
            if item.get("manifest_recorded") else " — not present in manifest['artifacts'] either"
        )
        + "</li>"
        for item in health.get("missing_detail", [])
    )
    stability_rows = "".join([
        f"<tr><td>Epochs recorded</td><td>{_fmt(stability.get('epochs'))}</td></tr>",
        f"<tr><td>Best validation positive Dice</td><td>{_fmt(stability.get('best_dice'))}</td></tr>",
        f"<tr><td>Best epoch</td><td>{_fmt(stability.get('best_epoch'))}</td></tr>",
        f"<tr><td>Train loss drop</td><td>{_fmt(stability.get('loss_drop'))}</td></tr>",
        f"<tr><td>Validation Dice std</td><td>{_fmt(stability.get('dice_std'))}</td></tr>",
        f"<tr><td>Validation IoU std</td><td>{_fmt(stability.get('iou_std'))}</td></tr>",
        f"<tr><td>Dice direction changes</td><td>{_fmt(stability.get('oscillations'))}</td></tr>",
        f"<tr><td>Learning rate</td><td>{_fmt(stability.get('learning_rate_start'))} → {_fmt(stability.get('learning_rate_end'))}</td></tr>",
    ]) if stability.get("available") else "<tr><td colspan='2'>No training history available.</td></tr>"

    error_rows = ""
    if errors.get("available"):
        error_rows = "".join([
            f"<tr><td>TP</td><td>{errors['tp']:.1f}</td></tr>",
            f"<tr><td>FN</td><td>{errors['fn']:.1f}</td></tr>",
            f"<tr><td>TN</td><td>{errors['tn']:.1f}</td></tr>",
            f"<tr><td>FP</td><td>{errors['fp']:.1f}</td></tr>",
            f"<tr><td>Prediction records</td><td>{_fmt(errors.get('records'))}</td></tr>" if errors.get('records') is not None else "",
            f"<tr><td colspan='2' class='muted'>{_esc(errors['note'])}</td></tr>",
        ])
    else:
        error_rows = "<tr><td colspan='2'>Image-level confusion profile unavailable.</td></tr>"

    audit_rows = "".join([
        f"<tr><td>Status</td><td>{_esc(audit.get('status', 'UNKNOWN'))}</td></tr>",
        f"<tr><td>Source</td><td>{_esc(audit.get('path', '—'))}</td></tr>",
        f"<tr><td>Placeholder markers</td><td>{_esc(audit.get('placeholder_hits', '—'))}</td></tr>",
        f"<tr><td>Conv2d occurrences</td><td>{_esc(audit.get('conv2d_count', '—'))}</td></tr>",
        f"<tr><td>Forward method</td><td>{'yes' if audit.get('has_forward') else 'no'}</td></tr>",
        f"<tr><td>Attention-like code</td><td>{'yes' if audit.get('has_attention_like_code') else 'no'}</td></tr>",
        f"<tr><td>Transformer-like code</td><td>{'yes' if audit.get('has_transformer_like_code') else 'no'}</td></tr>",
        f"<tr><td>Actual runtime class</td><td>{_esc(signature.get('class', '—'))}</td></tr>",
        f"<tr><td>Parameter count</td><td>{_fmt(signature.get('parameter_count'))}</td></tr>",
        f"<tr><td>Trainable parameter count</td><td>{_fmt(signature.get('trainable_parameter_count'))}</td></tr>",
        f"<tr><td>Initial state hash</td><td><code>{_esc(signature.get('initial_state_hash', '—'))}</code></td></tr>",
        f"<tr><td>Source hash</td><td><code>{_esc((trace.get('source_fingerprints') or {}).get(f'models/{model}.py', '—'))}</code></td></tr>",
        f"<tr><td colspan='2' class='muted'>{_esc(audit.get('note', ''))}</td></tr>",
    ])

    explain_rows = "".join([
        f"<tr><td>Prediction/GT/overlay ready</td><td>{'yes' if explain.get('base_visuals_ready') else 'no'}</td></tr>",
        f"<tr><td>Grad-CAM generated</td><td>{'yes' if explain.get('gradcam_available') else 'no'}</td></tr>",
        f"<tr><td>Layer hooks</td><td>{'yes' if explain.get('layer_hooks_available') else 'no'}</td></tr>",
        f"<tr><td>Attention signal in source</td><td>{'yes' if explain.get('attention_source_signal') else 'no'}</td></tr>",
        f"<tr><td colspan='2' class='muted'>{_esc(explain.get('note', ''))}</td></tr>",
    ])

    return f'''<section class="card"><h2>Diagnostics</h2>
<p class="muted">Run {_esc(run_id)} · {_esc(model)} · status {_esc(status)}</p>
<div class="diag-tabs"><a href="#overview">Overview</a><a href="#stability">Training Stability</a><a href="#errors">Error Analysis</a><a href="#audit">Research Trace</a><a href="#architecture">Architecture Audit</a><a href="#explainability">Explainability</a></div>
<section id="overview"><h3>Automatic Result Checks</h3><ul>{warning_rows}</ul><p><strong>Artifact health:</strong> {health['existing']}/{health['total']} required artifacts present; {health['missing']} missing.</p>{f'<ul class="muted">{missing_detail_rows}</ul>' if missing_detail_rows else ''}</section>
<section id="stability"><h3>Training Stability</h3><table class="data-table"><tbody>{stability_rows}</tbody></table></section>
<section id="errors"><h3>Error Analysis</h3><table class="data-table"><tbody>{error_rows}</tbody></table><p class="muted">When test_predictions.csv exists, this profile is computed directly from persisted per-image records; otherwise the dashboard falls back to an approximate profile from persisted recall/specificity.</p></section>
<section id="audit"><h3>Research Trace &amp; Verification</h3><table class="data-table"><tbody>
<tr><td>Metric completeness</td><td>{_fmt(len(load_json(paths.get("metric_status", Path(""))))) if paths.get("metric_status") else "—"}</td></tr>
<tr><td>Test prediction records</td><td>{_fmt(len(case_records))}</td></tr>
<tr><td>Training-history duplicate</td><td>{"yes" if (verification.get("cross_run_duplicates") or {}).get("duplicate_training_histories") else "no"}</td></tr>
<tr><td>Prediction-fingerprint duplicate</td><td>{"yes" if (verification.get("cross_run_duplicates") or {}).get("duplicate_prediction_records") else "no"}</td></tr>
<tr><td>Independent verification</td><td>{_esc(verification.get("status", "NOT_AVAILABLE"))}</td></tr>
</tbody></table><p class="muted">The run records runtime model identity, configuration, dataset fingerprints, RNG state, checkpoint hashes, per-image predictions and independent metric verification.</p></section>
<section id="architecture"><h3>Architecture Audit</h3><table class="data-table"><tbody>{audit_rows}</tbody></table></section>
<section id="explainability"><h3>Explainability Readiness</h3><table class="data-table"><tbody>{explain_rows}</tbody></table></section>
</section>'''


def _esc(value):
    import html
    return html.escape(str(value))
