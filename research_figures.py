"""Generate manuscript-facing figures from persisted MedPlug run artifacts.

All plotted measurements are read from run-local audit files. The dashboard does
not fabricate curves, substitute placeholder scores, or silently mix the official
and positive-only diagnostic protocols.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PAPER_MODELS = ("attention_unet", "transunet", "segformer")
MODEL_LABELS = {
    "attention_unet": "Attention U-Net",
    "transunet": "TransUNet-style",
    "segformer": "SegFormer-style",
}
FIGURE_DIR = Path("experiments") / "research_figures"
CACHE_NAME = "qualitative_cache.npz"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, UnicodeError):
        pass
    return rows


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _run_candidates(project_root: Path) -> dict[str, dict]:
    """Find the latest completed diagnostic run for each paper model.

    A candidate must have persisted per-image predictions for a positive-only
    174-image test split and an epoch diagnostic log. This intentionally keeps
    the official-split U-Net run out of the three-model diagnostic comparison.
    """
    experiments = project_root / "experiments"
    candidates: dict[str, list[dict]] = {model: [] for model in PAPER_MODELS}
    if not experiments.is_dir():
        return {}

    for run_dir in experiments.glob("RUN_*"):
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        status = str(manifest.get("status", "")).upper()
        model = str(manifest.get("model", "")).lower()
        if model not in candidates or not status.startswith("COMPLETED"):
            continue

        config = manifest.get("configuration") if isinstance(manifest.get("configuration"), dict) else {}
        if _to_int(config.get("input_channels"), 1) != 1:
            continue
        test_rows = _read_csv(run_dir / "audit" / "test_predictions.csv")
        epochs = _read_jsonl(run_dir / "audit" / "epoch_diagnostics.jsonl")
        if len(test_rows) != 174 or not epochs:
            continue
        if any(_to_int(row.get("GroundTruthPositive")) != 1 for row in test_rows):
            continue
        if not all(str(row.get("ImageId", "")).strip() for row in test_rows):
            continue

        # Keep the runs with the positive-only diagnostic configuration, rather
        # than similarly sized experiments whose protocol is not recorded.
        diagnostic_paths = " ".join(
            str(config.get(key, "")) for key in ("train_csv_rel", "val_csv_rel", "test_csv_rel")
        ).lower()
        if "diagnostic" not in diagnostic_paths and "aligned" not in diagnostic_paths:
            continue

        completed_at = str(manifest.get("completed_at", ""))
        candidates[model].append({
            "run_id": str(manifest.get("run_id") or run_dir.name),
            "model": model,
            "label": MODEL_LABELS[model],
            "root": run_dir,
            "manifest": manifest,
            "config": config,
            "test_rows": test_rows,
            "validation_rows": _read_csv(run_dir / "audit" / "validation_predictions.csv"),
            "epochs": epochs,
            "completed_at": completed_at,
            "sort_key": completed_at or f"{run_dir.stat().st_mtime_ns:020d}",
        })

    selected = {}
    for model, runs in candidates.items():
        if runs:
            selected[model] = sorted(runs, key=lambda row: row["sort_key"])[-1]
    return selected


def _history_values(run: dict) -> tuple[list[int], list[float]]:
    epochs, values = [], []
    for row in run["epochs"]:
        epoch = _to_int(row.get("epoch"), len(epochs) + 1)
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        positive = validation.get("positive") if isinstance(validation.get("positive"), dict) else {}
        value = _to_float(positive.get("dice"))
        if value is None:
            value = _to_float(row.get("ValidationDicePositive"))
        if value is not None:
            epochs.append(epoch)
            values.append(value)
    return epochs, values


def _metrics_by_id(run: dict) -> dict[str, dict]:
    output = {}
    for row in run["test_rows"]:
        image_id = str(row.get("ImageId", "")).strip()
        dice = _to_float(row.get("Dice"))
        if image_id and dice is not None:
            output[image_id] = {
                "dice": dice,
                "positive": _to_int(row.get("GroundTruthPositive")) == 1,
            }
    return output


def _validation_ids(run: dict) -> set[str]:
    return {
        str(row.get("ImageId", "")).strip()
        for row in run["validation_rows"]
        if str(row.get("ImageId", "")).strip()
    }


def _bootstrap_interval(values: np.ndarray, seed: int = 42, samples: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def save_qualitative_cache(path: Path, visual_samples: list[dict]) -> None:
    """Persist inference-time image, target and prediction arrays compactly.

    Masks are bit-packed to keep audit storage small. This cache is created from
    the same forward pass used for the per-image metric CSV, not from a second
    scoring pass after the test evaluation.
    """
    samples = [item for item in visual_samples if isinstance(item, dict) and item.get("image_id")]
    if not samples:
        return

    ids = np.asarray([str(item["image_id"]) for item in samples], dtype=str)
    images = np.stack([np.asarray(item["image"], dtype=np.uint8) for item in samples], axis=0)
    ground_truth = np.stack([np.asarray(item["ground_truth"], dtype=np.uint8) for item in samples], axis=0)
    predictions = np.stack([np.asarray(item["prediction"], dtype=np.uint8) for item in samples], axis=0)
    if images.ndim != 3 or ground_truth.shape != images.shape or predictions.shape != images.shape:
        raise ValueError("Qualitative cache expects same-sized grayscale images and masks.")
    height, width = images.shape[1:]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_ids=ids,
        images=images,
        ground_truth_packed=np.packbits(ground_truth.reshape(len(samples), -1), axis=1),
        predictions_packed=np.packbits(predictions.reshape(len(samples), -1), axis=1),
        image_shape=np.asarray([height, width], dtype=np.int32),
    )


def _load_cache(path: Path) -> dict[str, dict] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = [str(value) for value in archive["image_ids"].tolist()]
            images = archive["images"]
            height, width = [int(value) for value in archive["image_shape"].tolist()]
            bit_count = height * width
            ground = np.unpackbits(archive["ground_truth_packed"], axis=1, count=bit_count).reshape(-1, height, width)
            preds = np.unpackbits(archive["predictions_packed"], axis=1, count=bit_count).reshape(-1, height, width)
            if len(ids) != len(images) or len(ids) != len(ground) or len(ids) != len(preds):
                return None
            return {
                image_id: {
                    "image": images[index],
                    "ground_truth": ground[index],
                    "prediction": preds[index],
                }
                for index, image_id in enumerate(ids)
            }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _source_signature(runs: dict[str, dict]) -> str:
    source_paths = []
    for run in runs.values():
        root = run["root"]
        source_paths.extend([
            root / "manifest.json",
            root / "audit" / "epoch_diagnostics.jsonl",
            root / "audit" / "test_predictions.csv",
            root / "audit" / "validation_predictions.csv",
            root / "audit" / CACHE_NAME,
        ])
    bits = []
    for path in sorted(set(source_paths), key=lambda value: str(value)):
        try:
            stat = path.stat()
            bits.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            bits.append(f"{path}:missing")
    return hashlib.sha256("\n".join(bits).encode("utf-8")).hexdigest()


def _make_figures(project_root: Path, runs: dict[str, dict], out_dir: Path) -> dict:
    # Import plotting only when figures need to be generated, not at app startup.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Fig. 2: bootstrap intervals from the same shared test IDs, with overlap
    # exposed rather than hidden. The disjoint estimate drops every ID appearing
    # in any selected run's validation list.
    if runs:
        ids_by_model = {model: set(_metrics_by_id(run)) for model, run in runs.items()}
        common_ids = sorted(set.intersection(*ids_by_model.values())) if ids_by_model else []
        shared_validation = set().union(*(_validation_ids(run) for run in runs.values()))
        disjoint_ids = [image_id for image_id in common_ids if image_id not in shared_validation]
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        x_positions = np.arange(len(runs), dtype=float)
        width = 0.34
        plotted = False
        for offset, (subset_label, subset_ids) in enumerate((
            ("All shared test cases", common_ids),
            ("Validation-disjoint cases", disjoint_ids),
        )):
            means, lows, highs = [], [], []
            for model in runs:
                metric_map = _metrics_by_id(runs[model])
                values = np.asarray([metric_map[i]["dice"] for i in subset_ids if i in metric_map and metric_map[i]["positive"]], dtype=float)
                mean, low, high = _bootstrap_interval(values, seed=42 + offset)
                means.append(mean)
                lows.append(max(0.0, mean - low) if math.isfinite(mean) else np.nan)
                highs.append(max(0.0, high - mean) if math.isfinite(mean) else np.nan)
            shift = (offset - 0.5) * width
            pos = x_positions + shift
            if any(math.isfinite(value) for value in means):
                ax.bar(pos, means, width=width, label=f"{subset_label} (n={len(subset_ids)})", alpha=0.86)
                ax.errorbar(pos, means, yerr=np.asarray([lows, highs]), fmt="none", ecolor="#111827", capsize=4, linewidth=1.2)
                plotted = True
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions, [run["label"] for run in runs.values()], rotation=12, ha="right")
        ax.set_ylabel("Mean per-image Dice")
        ax.set_title("Test Dice with 95% Case-Level Bootstrap Intervals")
        ax.grid(axis="y", alpha=0.22)
        if plotted:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No shared per-image test metrics available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        out = out_dir / "figure2_test_dice_ci.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["figure2"] = out

    # Fig. 3: one axis, actual positive-case validation Dice values from epoch logs.
    has_history = any(_history_values(run)[0] for run in runs.values())
    if runs and has_history:
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        for run in runs.values():
            epochs, dice = _history_values(run)
            if epochs:
                ax.plot(epochs, dice, marker="o", markersize=3, linewidth=1.7, label=run["label"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Positive-case validation Dice")
        ax.set_ylim(0, 1)
        ax.set_title("Positive-Case Validation Dice by Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        out = out_dir / "figure3_validation_dice_by_epoch.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["figure3"] = out

    # Fig. 4: choose high, median and low cases on the *same* test IDs across
    # every model; exclude anything in any selected validation set.
    if len(runs) == len(PAPER_MODELS):
        id_sets = [set(_metrics_by_id(run)) for run in runs.values()]
        common_ids = set.intersection(*id_sets) if id_sets else set()
        val_ids = set().union(*(_validation_ids(run) for run in runs.values()))
        common_ids = sorted(image_id for image_id in common_ids if image_id not in val_ids and all(_metrics_by_id(run).get(image_id, {}).get("positive", False) for run in runs.values()))
        caches = {model: _load_cache(run["root"] / "audit" / CACHE_NAME) for model, run in runs.items()}
        cache_ready = all(cache is not None for cache in caches.values())
        if cache_ready and len(common_ids) >= 3:
            avg_dice = {}
            for image_id in common_ids:
                avg_dice[image_id] = float(np.mean([_metrics_by_id(run)[image_id]["dice"] for run in runs.values()]))
            ranked_ids = sorted(common_ids, key=lambda image_id: (avg_dice[image_id], image_id))
            selected_ids = [ranked_ids[-1], ranked_ids[len(ranked_ids) // 2], ranked_ids[0]]

            # Confirm the same ImageId really maps to identical pixels/mask in every run.
            reference_model = next(iter(runs))
            pixel_mismatch = False
            for image_id in selected_ids:
                reference = caches[reference_model].get(image_id)
                if reference is None:
                    pixel_mismatch = True
                    break
                for model in runs:
                    sample = caches[model].get(image_id)
                    if sample is None or not np.array_equal(sample["image"], reference["image"]) or not np.array_equal(sample["ground_truth"], reference["ground_truth"]):
                        pixel_mismatch = True
                        break
                if pixel_mismatch:
                    break

            if pixel_mismatch:
                outputs["figure4_pending"] = {
                    "cache_ready": True,
                    "shared_disjoint_cases": len(common_ids),
                    "runs_missing_cache": [],
                    "reason": "Matching ImageIds have different image or ground-truth pixels between runs; no qualitative grid was emitted.",
                }
            else:
                case_labels = ["Best mean Dice", "Median mean Dice", "Lowest mean Dice"]
                fig, axes = plt.subplots(3, 2 + len(runs), figsize=(3.05 * (2 + len(runs)), 9.0))
                for row_index, (image_id, case_label) in enumerate(zip(selected_ids, case_labels)):
                    reference = caches[reference_model][image_id]
                    axes[row_index, 0].imshow(reference["image"], cmap="gray")
                    axes[row_index, 0].set_title(f"X-ray\n{case_label}", fontsize=9)
                    axes[row_index, 0].set_ylabel(f"{case_label}\nDice={avg_dice[image_id]:.3f}", fontsize=9)
                    axes[row_index, 1].imshow(reference["ground_truth"], cmap="gray", vmin=0, vmax=1)
                    axes[row_index, 1].set_title("Ground truth", fontsize=9)
                    for col, model in enumerate(runs, start=2):
                        sample = caches[model][image_id]
                        axes[row_index, col].imshow(sample["image"], cmap="gray")
                        prediction = np.asarray(sample["prediction"], dtype=bool)
                        if prediction.any():
                            axes[row_index, col].imshow(prediction, cmap="Reds", alpha=0.42, vmin=0, vmax=1)
                        gt = np.asarray(sample["ground_truth"], dtype=bool)
                        if gt.any():
                            axes[row_index, col].contour(gt.astype(float), levels=[0.5], colors=["lime"], linewidths=0.7)
                        case_metric = _metrics_by_id(runs[model])[image_id]["dice"]
                        axes[row_index, col].set_title(f"{runs[model]['label']}\nDice={case_metric:.3f}", fontsize=9)
                    for col in range(2 + len(runs)):
                        axes[row_index, col].axis("off")
                fig.suptitle("Qualitative Test Cases — Shared Validation-Disjoint IDs", fontsize=13, y=1.01)
                fig.tight_layout()
                out = out_dir / "figure4_qualitative_grid.png"
                fig.savefig(out, dpi=180, bbox_inches="tight")
                plt.close(fig)
                outputs["figure4"] = out
        else:
            outputs["figure4_pending"] = {
                "cache_ready": cache_ready,
                "shared_disjoint_cases": len(common_ids),
                "runs_missing_cache": [MODEL_LABELS[model] for model, cache in caches.items() if cache is None],
            }

    return outputs


def _figure_html(title: str, path: Path, project_root: Path, caption: str) -> str:
    rel = path.relative_to(project_root).as_posix()
    return (
        '<section class="card"><h3>' + title + '</h3>'
        f'<img src="/{rel}" alt="{title}" loading="lazy" style="display:block;width:100%;max-width:1100px;height:auto;border:1px solid #dbe3ec;border-radius:10px;margin:10px auto">'
        f'<p class="note">{caption}</p></section>'
    )


def render_research_figures(project_root: Path) -> str:
    """Generate figures if artifacts changed, then return dashboard HTML."""
    project_root = Path(project_root).resolve()
    out_dir = project_root / FIGURE_DIR
    runs = _run_candidates(project_root)
    signature = _source_signature(runs)
    signature_path = out_dir / "source_signature.json"
    previous = _read_json(signature_path)
    existing_outputs = [out_dir / name for name in (
        "figure2_test_dice_ci.png",
        "figure3_validation_dice_by_epoch.png",
        "figure4_qualitative_grid.png",
    )]
    required_missing = any(not path.is_file() for path in (existing_outputs[0], existing_outputs[1]) if runs)
    if previous.get("signature") != signature or required_missing:
        outputs = _make_figures(project_root, runs, out_dir)
        # Remove stale figures so the dashboard never shows plots from older runs.
        for key, filename in (("figure2", "figure2_test_dice_ci.png"), ("figure3", "figure3_validation_dice_by_epoch.png"), ("figure4", "figure4_qualitative_grid.png")):
            if key not in outputs:
                (out_dir / filename).unlink(missing_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = outputs.get("figure4_pending", {})
        signature_path.write_text(json.dumps({"signature": signature, "figure4_pending": pending}, indent=2), encoding="utf-8")
    else:
        pending = previous.get("figure4_pending", {})

    cards = []
    fig2 = out_dir / "figure2_test_dice_ci.png"
    fig3 = out_dir / "figure3_validation_dice_by_epoch.png"
    fig4 = out_dir / "figure4_qualitative_grid.png"
    if fig2.is_file():
        cards.append(_figure_html(
            "Figure 2 — Test Dice and 95% Bootstrap Intervals", fig2, project_root,
            "Computed from persisted per-image test Dice for the shared positive-only test IDs. The second series removes any test ImageId found in any selected model's validation CSV. Each interval uses 5,000 case-level bootstrap resamples; it does not estimate seed-to-seed variation.",
        ))
    if fig3.is_file():
        cards.append(_figure_html(
            "Figure 3 — Positive-Case Validation Dice by Epoch", fig3, project_root,
            "Read directly from each run's audit/epoch_diagnostics.jsonl. Curves are not smoothed and missing epochs are not interpolated.",
        ))
    if fig4.is_file():
        cards.append(_figure_html(
            "Figure 4 — Qualitative Test Examples", fig4, project_root,
            "Three shared positive test cases (best, median and lowest mean Dice across the three models), selected only after excluding ImageIds present in any selected validation split. Predictions are overlays from the stored inference-time cache; lime contours mark the ground-truth boundary.",
        ))
    else:
        missing_cache = pending.get("runs_missing_cache", [])
        reason = pending.get("reason")
        if reason:
            names = "the pixel-consistency check"
        else:
            names = ", ".join(missing_cache) if missing_cache else "one or more of the three model runs"
        count = pending.get("shared_disjoint_cases")
        extra = f" Shared validation-disjoint positive test cases currently available: {count}." if count is not None else ""
        reason_html = f" Reason: {reason}." if reason else ""
        cards.append(
            '<section class="card"><h3>Figure 4 — Qualitative Test Examples</h3>'
            '<div class="warning"><strong>Not generated yet.</strong> The current run CSVs preserve per-image metrics and hashes, but not the input pixels and mask arrays required to reconstruct a truthful cross-model image grid. '
            f'Missing inference-time image/mask cache for: {names}.{extra}{reason_html} '
            'First try `python tools/build_qualitative_cache.py` from the project root to backfill existing runs. It writes a cache only when the saved checkpoint and dataset reproduce every archived prediction-mask hash and per-image Dice value. If verification fails or the source data/checkpoint is unavailable, retrain the three diagnostic models with this updated project and configured dataset root. New runs save the cache during test inference, and the figure appears here automatically.</div></section>'
        )

    if runs:
        shared_val = set().union(*(_validation_ids(run) for run in runs.values()))
        rows = []
        for model in PAPER_MODELS:
            run = runs.get(model)
            if not run:
                rows.append(f'<tr><td>{MODEL_LABELS[model]}</td><td colspan="4">No completed diagnostic run found</td></tr>')
                continue
            test_ids = set(_metrics_by_id(run))
            overlap = len(test_ids & _validation_ids(run))
            verification = _read_json(run["root"] / "audit" / "verification.json")
            verify_status = str(verification.get("status", "NOT AVAILABLE"))
            cache_exists = (run["root"] / "audit" / CACHE_NAME).is_file()
            rows.append(
                f'<tr><td>{run["label"]}</td><td>{run["run_id"]}</td><td>{len(test_ids)}</td>'
                f'<td>{overlap}</td><td><span class="status-pill">{verify_status}</span></td>'
                f'<td>{"available" if cache_exists else "missing"}</td></tr>'
            )
        audit_table = (
            '<section class="card"><h3>Evidence available from the selected diagnostic runs</h3>'
            '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Model</th><th>Run ID</th><th>Test cases</th><th>Test/validation ID overlap</th><th>Metric verification</th><th>Pixel cache</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
            '<p class="note">The official-split U-Net run is intentionally not ranked in these figures because it uses a different split/protocol. A failed verification remains visible; plots do not override audit status.</p></section>'
        )
    else:
        audit_table = (
            '<section class="card"><h3>Research figures</h3><p class="muted">No completed diagnostic runs were found. Complete the attention_unet, transunet and segformer diagnostic runs to populate manuscript figures.</p></section>'
        )

    return audit_table + "".join(cards)


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    render_research_figures(project)
    print(f"Research figure dashboard artifacts: {project / FIGURE_DIR}")
