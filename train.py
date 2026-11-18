import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from eval import evaluate
from experiment_manager import ExperimentManager
from core.plugin_loader import load_config_yaml, load_dataset_plugin, load_model_plugin, load_preprocess_plugin
from dataset_contract import validate_contract
from report_generator import ReportGenerator
from pipeline import run_post_training_pipeline
from utils import DiceBCELoss
from research_trace import (
    capture_rng_state,
    extract_model_state,
    save_initial_trace,
    save_rich_checkpoint,
    save_rng_state,
    save_epoch_trace,
    write_audit_bundle,
    write_json,
)
from visualization import Visualizer
from research_figures import save_qualitative_cache


def configure_console_encoding():
    """Keep Windows console/file output from failing on Unicode characters."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model on the SIIM-ACR dataset.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIRECTORY)
    parser.add_argument("--train-dicom-dir", type=Path, default=config.DICOM_TRAIN_DIRECTORY)
    parser.add_argument("--test-dicom-dir", type=Path, default=config.DICOM_TEST_DIRECTORY)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--loss", choices=["bce", "dice_bce"], default="dice_bce")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=config.DEVICE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--model", choices=["unet", "attention_unet", "transunet", "segformer"], default=config.MODEL)
    parser.add_argument("--pretrained", action="store_true", default=config.PRETRAINED)
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=config.INPUT_CHANNELS)
    parser.add_argument("--output-channels", type=int, default=config.OUTPUT_CHANNELS)
    parser.add_argument("--normalization", choices=["minmax", "imagenet"], default=config.NORMALIZATION)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIRECTORY)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--early-stopping", action="store_true", default=config.EARLY_STOPPING_ENABLED)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--resume", action="store_true", default=config.RESUME_TRAINING, help="Resume from the latest completed-epoch checkpoint for --run-id.")
    parser.add_argument("--run-id", type=str, default=None, help="Existing run ID when resuming; otherwise a new run is created.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="YAML plugin configuration.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Optional evaluation metrics to include in the report.")
    return parser.parse_args()


def apply_yaml_config(args):
    yaml_data = load_config_yaml(args.config)
    dataset_cfg = yaml_data.get("dataset", {})
    preprocess_cfg = yaml_data.get("preprocess", {})
    model_cfg = yaml_data.get("model", {})
    training_cfg = yaml_data.get("training", {})

    if dataset_cfg.get("plugin") != "siim_acr":
        raise ValueError(f"Unsupported dataset plugin: {dataset_cfg.get('plugin')}")
    preprocess_plugin_name = preprocess_cfg.get("plugin", "segmentation_default")
    prep = load_preprocess_plugin(preprocess_plugin_name, preprocess_config=preprocess_cfg)

    root_value = os.path.expandvars(os.path.expanduser(str(dataset_cfg["root"])))
    if not root_value or root_value.startswith("${"):
        raise FileNotFoundError(
            "dataset.root is not configured. Set MEDPLUG_DATASET_ROOT in PowerShell or provide a dataset root in the YAML config."
        )
    args.data_root = Path(root_value).resolve()
    args.dataset_plugin = dataset_cfg["plugin"]
    args.train_csv_rel = dataset_cfg.get("train_csv", "data/train-rle.csv")
    args.val_csv_rel = dataset_cfg.get("val_csv", "data/val-rle.csv")
    args.test_csv_rel = dataset_cfg.get("test_csv", "data/test-rle.csv")
    args.train_images_rel = dataset_cfg.get("train_images", "images/dicom-images-train")
    args.val_images_rel = dataset_cfg.get("val_images", args.train_images_rel)
    args.test_images_rel = dataset_cfg.get("test_images", "images/dicom-images-test")
    args.aligned_images_rel = dataset_cfg.get("aligned_images", "aligned_dataset/images")
    args.aligned_masks_rel = dataset_cfg.get("aligned_masks", "aligned_dataset/masks")
    args.source_mode = str(dataset_cfg.get("source_mode", "auto"))
    args.strict_counts = bool(dataset_cfg.get("strict_counts", True))
    args.model = model_cfg.get("plugin", args.model)
    args.pretrained = bool(model_cfg.get("pretrained", args.pretrained))
    args.input_channels = prep.input_channels
    args.normalization = prep.normalization
    config.IMAGE_SIZE = tuple(prep.image_size)
    args.loss = str(training_cfg.get("loss", args.loss))
    # An explicit CLI --resume must override the YAML value. Otherwise a YAML
    # file with resume: false silently turns off a requested resume run.
    cli_resume_requested = "--resume" in sys.argv[1:]
    args.resume = cli_resume_requested or bool(training_cfg.get("resume", args.resume))
    args.metrics = list(yaml_data.get("evaluation", {}).get("metrics", args.metrics or ["dice", "iou", "hd95"]))

    for key, attr, caster in [
        ("batch_size", "batch_size", int),
        ("num_workers", "num_workers", int),
        ("epochs", "epochs", int),
        ("learning_rate", "lr", float),
        ("random_seed", "seed", int),
        ("device", "device", str),
        ("early_stopping_enabled", "early_stopping", bool),
        ("patience", "patience", int),
    ]:
        if key in training_cfg:
            setattr(args, attr, caster(training_cfg[key]))
    for key in ("weight_decay", "optimizer", "scheduler", "gradient_accumulation_steps", "segmentation_threshold"):
        if key in training_cfg:
            if key == "weight_decay":
                config.WEIGHT_DECAY = float(training_cfg[key])
            elif key == "optimizer":
                config.OPTIMIZER = str(training_cfg[key])
            elif key == "scheduler":
                config.SCHEDULER = str(training_cfg[key])
            elif key == "gradient_accumulation_steps":
                config.GRADIENT_ACCUMULATION_STEPS = int(training_cfg[key])
            elif key == "segmentation_threshold":
                config.SEGMENTATION_THRESHOLD = float(training_cfg[key])
    return yaml_data



def count_split_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)
    positive = sum(
        any(str(rle).strip() not in {"", "-1", "nan"} for rle in rles)
        for rles in grouped
    )
    total = len(grouped)
    return total, positive, total - positive


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Refusing CPU fallback.")
    return selected


def print_split_summary(name, csv_path, dataset, dicom_dir):
    total, positive, negative = count_split_labels(csv_path)
    source_counts = getattr(dataset, "source_counts", {})
    missing = max(total - int(source_counts.get("total", len(dataset))), 0)
    print(f"{name} split:")
    print(f"  csv file: {csv_path}")
    print(f"  unique images in CSV: {total}")
    print(f"  loaded images: {len(dataset)}")
    print(f"  positive mask images: {positive}")
    print(f"  negative mask images: {negative}")
    print(f"  aligned image/mask source: {source_counts.get('aligned', 0)}")
    print(f"  DICOM source: {source_counts.get('dicom', 0)}")
    print(f"  missing image sources: {missing}")
    if missing:
        raise RuntimeError(f"{name}: {missing} image sources are missing; refusing to start training.")
    print()
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "loaded": len(dataset),
        "source_counts": source_counts,
    }


def train_one_epoch(model, loader, device, optimizer, criterion, accumulation_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batch_count = 0
    grad_norms = []
    target_fraction = []
    prediction_fraction = []
    probability_means = []
    logit_means = []

    for batch_idx, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"Non-finite model output detected at training batch {batch_idx}.")

        probabilities = torch.sigmoid(outputs)
        loss = criterion(outputs, masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss detected at batch {batch_idx}.")
        (loss / accumulation_steps).backward()

        should_step = batch_idx % accumulation_steps == 0 or batch_idx == len(loader)
        if should_step:
            grad_sq = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite gradient detected at training batch {batch_idx}.")
                grad_sq = grad_sq + parameter.grad.detach().norm(2).pow(2)
            grad_norms.append(float(torch.sqrt(grad_sq).item()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        batch_count += 1
        target_fraction.append(float(masks.mean().item()))
        prediction_fraction.append(float((probabilities >= config.SEGMENTATION_THRESHOLD).float().mean().item()))
        probability_means.append(float(probabilities.mean().item()))
        logit_means.append(float(outputs.mean().item()))

        if batch_idx % 100 == 0:
            print(f"    batch {batch_idx}/{len(loader)}, loss: {loss.item():.6f}")

    def avg(values):
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "loss": total_loss / max(batch_count, 1),
        "target_positive_fraction": avg(target_fraction),
        "prediction_positive_fraction": avg(prediction_fraction),
        "probability_mean": avg(probability_means),
        "probability_std": float(np.std(probability_means)) if probability_means else 0.0,
        "output_mean": avg(logit_means),
        "output_std": float(np.std(logit_means)) if logit_means else 0.0,
        "grad_norm_mean": avg(grad_norms),
        "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
        "optimizer_steps": len(grad_norms),
    }


def find_visualization_sample(dataset, loader, device, model):
    for images, masks in loader:
        positive = masks.sum(dim=(1, 2, 3)) > 0
        index = int(torch.where(positive)[0][0]) if positive.any() else 0
        image = images[index:index + 1].to(device)
        mask = masks[index:index + 1].to(device)
        with torch.no_grad():
            output = model(image)
        return image, mask, output
    return None


def main():
    configure_console_encoding()
    args = parse_args()
    yaml_data = apply_yaml_config(args)

    train_csv = args.data_root / args.train_csv_rel
    val_csv = args.data_root / args.val_csv_rel
    test_csv = args.data_root / args.test_csv_rel
    train_dicom_dir = args.data_root / args.train_images_rel
    val_dicom_dir = args.data_root / args.val_images_rel
    test_dicom_dir = args.data_root / args.test_images_rel
    aligned_images_dir = args.data_root / args.aligned_images_rel
    aligned_masks_dir = args.data_root / args.aligned_masks_rel

    set_random_seed(args.seed)
    rng_before_model = capture_rng_state()

    validate_contract([
        ("TRAIN", train_csv, train_dicom_dir, aligned_images_dir, aligned_masks_dir),
        ("VALIDATION", val_csv, val_dicom_dir, aligned_images_dir, aligned_masks_dir),
        ("TEST", test_csv, test_dicom_dir, aligned_images_dir, aligned_masks_dir),
    ], source_mode=args.source_mode, strict_counts=args.strict_counts)

    if args.checkpoint_name is None:
        args.checkpoint_name = f"{args.model}_best_model"
    experiment_manager = ExperimentManager()
    experiment_paths = experiment_manager.create_experiment(
        args.model,
        args.checkpoint_name,
        resume=args.resume,
        run_id=args.run_id,
        existing=bool(args.run_id and not args.resume),
    )
    args.run_id = experiment_paths.run_id
    if args.resume and not experiment_paths.resume_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume requested, but no resume checkpoint exists for run {args.run_id}: "
            f"{experiment_paths.resume_checkpoint_path}"
        )

    dataset_factory = load_dataset_plugin(args.dataset_plugin, dataset_config=yaml_data["dataset"] | {"root": args.data_root})
    device = resolve_device(args.device)
    experiment_manager.initialize_artifacts(experiment_paths, args, device, reset=not args.resume)

    print("Loading datasets:")
    print(f"  dataset plugin: {args.dataset_plugin}")
    print(f"  dataset root: {args.data_root}")
    print(f"  train CSV: {train_csv}")
    print(f"  val CSV:   {val_csv}")
    print(f"  test CSV:  {test_csv}")
    print(f"  train DICOM dir: {train_dicom_dir}")
    print(f"  val DICOM dir:   {val_dicom_dir}")
    print(f"  test DICOM dir:  {test_dicom_dir}")
    print(f"  aligned images: {aligned_images_dir}")
    print(f"  aligned masks: {aligned_masks_dir}")
    print(f"  dataset source mode: {args.source_mode}")
    print(f"  strict official split counts: {args.strict_counts}")
    print(f"  model: {args.model}")
    print(f"  input channels: {args.input_channels}")
    print(f"  image size: {config.IMAGE_SIZE}")
    print(f"  normalization: {args.normalization}")
    print(f"  batch size: {args.batch_size}")
    print(f"  accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning rate: {args.lr}")
    print(f"  optimizer: {config.OPTIMIZER}")
    print(f"  scheduler: {config.SCHEDULER}")
    print(f"  random seed: {args.seed}")
    print(f"  device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"  run id: {experiment_paths.run_id}")
    print(f"  experiment dir: {experiment_paths.root}")
    print()

    train_dataset = dataset_factory.split("train", input_channels=args.input_channels, normalization=args.normalization, image_size=config.IMAGE_SIZE)
    val_dataset = dataset_factory.split("val", input_channels=args.input_channels, normalization=args.normalization, image_size=config.IMAGE_SIZE)
    test_dataset = dataset_factory.split("test", input_channels=args.input_channels, normalization=args.normalization, image_size=config.IMAGE_SIZE)

    def make_loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    test_loader = make_loader(test_dataset, False)

    summary = {
        "train": print_split_summary("Train", train_csv, train_dataset, train_dicom_dir),
        "val": print_split_summary("Validation", val_csv, val_dataset, val_dicom_dir),
        "test": print_split_summary("Test", test_csv, test_dataset, test_dicom_dir),
    }

    model = load_model_plugin(args.model, in_channels=args.input_channels, out_channels=args.output_channels, pretrained=args.pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Requested model: {args.model}")
    print(f"Actual model class: {model.__class__.__name__}")
    print(f"Model module: {model.__class__.__module__}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Trainable parameter count: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")

    criterion = torch.nn.BCEWithLogitsLoss() if args.loss == "bce" else DiceBCELoss(bce_weight=config.BCE_WEIGHT)
    print(f"Using loss: {criterion.__class__.__name__}")

    optimizer_name = config.OPTIMIZER.lower()
    if optimizer_name != "adamw":
        raise ValueError("This research baseline expects config.OPTIMIZER='adamw'.")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1)) if config.SCHEDULER.lower() == "cosine" else None

    metadata = save_initial_trace(
        run_root=experiment_paths.root, args=args, yaml_data=yaml_data, config_module=config, device=device,
        model=model, summary=summary, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        project_root=Path(__file__).resolve().parent, rng_state=rng_before_model,
    )

    best_val_dice = -1.0
    best_val_iou = 0.0
    best_val_results = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = args.epochs
    early_stopping_reason = "Completed all configured epochs."
    resume_elapsed = 0.0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(experiment_paths.resume_checkpoint_path, map_location=device, weights_only=False)
        expected_model = checkpoint.get("model_name")
        if expected_model and expected_model != args.model:
            raise RuntimeError(f"Resume checkpoint model '{expected_model}' does not match requested model '{args.model}'.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        best_val_iou = float(checkpoint.get("best_val_iou", 0.0))
        best_val_results = checkpoint.get("best_val_results")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resume_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"Resuming from completed epoch {start_epoch - 1}; next epoch: {start_epoch}")
        print(f"Best validation positive Dice so far: {best_val_dice:.6f} (epoch {best_epoch})")
        if start_epoch > args.epochs:
            print(
                "Resume checkpoint is already complete; "
                "skipping additional training and proceeding to evaluation/audit."
            )
            start_epoch = args.epochs + 1

    training_start = experiment_manager.start_timer()

    print(f"Starting training for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}:")
        train_stats = train_one_epoch(model, train_loader, device, optimizer, criterion, config.GRADIENT_ACCUMULATION_STEPS)
        val_results = evaluate(
            model,
            val_loader,
            device,
            threshold=config.SEGMENTATION_THRESHOLD,
            image_level_fraction=config.IMAGE_LEVEL_MIN_MASK_FRACTION,
        )
        print(f"  train loss: {train_stats['loss']:.6f}")
        print(f"  mean probability: {train_stats['probability_mean']:.6f}")
        print(f"  predicted positive fraction: {train_stats['prediction_positive_fraction']:.6f}")
        print(f"  mean gradient norm: {train_stats['grad_norm_mean']:.6f}")
        print(f"  val Dice (all): {val_results['all']['dice']:.6f}")
        print(f"  val Dice (positive): {val_results['positive']['dice']:.6f}")
        print(f"  val IoU (positive): {val_results['positive']['iou']:.6f}")

        elapsed_epoch = experiment_manager.elapsed_seconds(training_start)
        experiment_manager.append_training_log(
            experiment_paths,
            epoch,
            train_stats["loss"],
            val_results,
            optimizer.param_groups[0]["lr"],
            elapsed_epoch,
        )
        save_epoch_trace(
            experiment_paths.root,
            {
                "epoch": epoch,
                "train": train_stats,
                "validation": val_results,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed_epoch,
            },
        )

        current_metric = val_results["positive"]["dice"]
        if current_metric > best_val_dice:
            best_val_dice = current_metric
            best_val_iou = val_results["positive"]["iou"]
            best_val_results = val_results
            best_epoch = epoch
            epochs_without_improvement = 0
            save_rich_checkpoint(
                experiment_paths.checkpoint_path,
                model=model,
                metadata=metadata,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state={
                    "best_val_dice": best_val_dice,
                    "best_val_iou": best_val_iou,
                    "best_epoch": best_epoch,
                    "best_val_results": best_val_results,
                },
                rng_state=capture_rng_state(),
            )
            print(f"  Saved best checkpoint (rich): {experiment_paths.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping and epochs_without_improvement >= args.patience:
                stopped_epoch = epoch
                early_stopping_reason = f"No validation Dice improvement for {args.patience} consecutive epochs."
                print(f"  Early stopping: {early_stopping_reason}")
                break

        if scheduler is not None:
            scheduler.step()

        elapsed_now = resume_elapsed + experiment_manager.elapsed_seconds(training_start)
        experiment_manager.atomic_torch_save({
            "model_name": args.model,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_val_results": best_val_results,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed_now,
            "checkpoint_name": args.checkpoint_name,
            "run_id": args.run_id,
        }, experiment_paths.resume_checkpoint_path)
        print(f"  Saved resumable checkpoint: {experiment_paths.resume_checkpoint_path}")
        print()

    if best_val_results is None:
        raise RuntimeError("Training completed without a valid validation checkpoint.")

    # Load best checkpoint before final evaluation. Test is intentionally evaluated exactly once.
    loaded_best = torch.load(experiment_paths.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(extract_model_state(loaded_best))
    model.eval()

    experiment_manager.update_stage(experiment_paths, "EVALUATION", status="RUNNING")
    validation_forensic = evaluate(
        model,
        val_loader,
        device,
        threshold=config.SEGMENTATION_THRESHOLD,
        image_level_fraction=config.IMAGE_LEVEL_MIN_MASK_FRACTION,
        collect_cases=True,
    )
    validation_records = validation_forensic.pop("cases", [])

    test_start = time.perf_counter()
    test_visual_samples = []
    try:
        test_results = evaluate(
            model,
            test_loader,
            device,
            threshold=config.SEGMENTATION_THRESHOLD,
            image_level_fraction=config.IMAGE_LEVEL_MIN_MASK_FRACTION,
            collect_cases=True,
            collect_visual_samples=True,
        )
        test_visual_samples = test_results.pop("visual_samples", [])
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        experiment_manager.update_stage(experiment_paths, "EVALUATION", status="FAILED", error=message)
        raise
    test_time = time.perf_counter() - test_start
    test_records = test_results.pop("cases", [])
    if test_visual_samples:
        try:
            save_qualitative_cache(experiment_paths.root / "audit" / "qualitative_cache.npz", test_visual_samples)
            print(f"Saved inference-time qualitative cache: {experiment_paths.root / 'audit' / 'qualitative_cache.npz'}")
        except Exception as exc:
            # Research figures are optional post-run artifacts; never discard valid metrics because a cache write failed.
            print(f"WARNING: could not save qualitative cache: {type(exc).__name__}: {exc}")
    test_results["validation_dice"] = best_val_dice
    experiment_manager.update_stage(experiment_paths, "METRICS", status="RUNNING")

    training_time = resume_elapsed + experiment_manager.elapsed_seconds(training_start)

    # Persist evaluation metrics before optional visualization so a visualization failure cannot discard the benchmark result.
    try:
        experiment_manager.save_metrics(
            experiment_paths,
            best_validation_dice=best_val_dice,
            best_validation_iou=best_val_iou,
            validation_results=best_val_results,
            test_results=test_results,
            training_time=training_time,
            args=args,
            device=device,
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            early_stopping_reason=early_stopping_reason,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        experiment_manager.update_stage(experiment_paths, "METRICS", status="FAILED", error=message)
        raise

    history_rows = []
    if experiment_paths.training_log_path.is_file():
        import csv as _csv
        with experiment_paths.training_log_path.open("r", newline="", encoding="utf-8") as handle:
            history_rows = list(_csv.DictReader(handle))
    epoch_records = []
    epoch_path = experiment_paths.root / "audit" / "epoch_diagnostics.jsonl"
    if epoch_path.is_file():
        for line in epoch_path.read_text(encoding="utf-8").splitlines():
            try:
                epoch_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    save_rng_state(experiment_paths.root / "audit" / "rng_final.pt", capture_rng_state(), "final")

    # Keep the best checkpoint self-describing: the learned weights remain
    # unchanged, while the complete training history and final benchmark
    # summary are embedded beside the model state for later forensic review.
    best_payload = torch.load(experiment_paths.checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(best_payload, dict) and "model_state" in best_payload:
        best_payload["training_history"] = history_rows
        best_payload["final_test_results"] = test_results
        best_payload["best_validation_results"] = best_val_results
        best_payload["best_epoch"] = best_epoch
        best_payload["final_rng_saved"] = True
        experiment_manager.atomic_torch_save(best_payload, experiment_paths.checkpoint_path)

    write_audit_bundle(
        run_root=experiment_paths.root,
        metadata=metadata,
        test_results=test_results,
        validation_records=validation_records,
        test_records=test_records,
        history_rows=history_rows,
        epoch_records=epoch_records,
        metrics_path=experiment_paths.metrics_path,
        best_checkpoint_path=experiment_paths.checkpoint_path,
        resume_checkpoint_path=experiment_paths.resume_checkpoint_path,
        experiments_root=experiment_manager.experiments_root,
    )

    pipeline_result = run_post_training_pipeline(
        experiment_manager=experiment_manager,
        experiment_paths=experiment_paths,
        args=args,
        device=device,
        summary=summary,
        validation_results=best_val_results,
        test_results=test_results,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        training_time=training_time,
        test_time=test_time,
        early_stopping_reason=early_stopping_reason,
        sample_factory=lambda: find_visualization_sample(val_dataset, val_loader, device, model),
        visualizer=Visualizer,
        report_generator=ReportGenerator(),
    )

    experiment_manager.mark_completed(experiment_paths, stage="COMPLETED_WITH_WARNINGS" if pipeline_result.warnings else "COMPLETED")

    # Refresh the derived indexes only after the per-run audit and final run
    # status have been committed. A reporting failure should not invalidate a
    # successfully trained checkpoint or its run-local artifacts.
    try:
        from tools.rebuild_benchmark_index import rebuild as rebuild_benchmark_index
        rebuild_benchmark_index(experiment_manager.experiments_root.parent, write=True)
    except Exception as exc:
        print(f"WARNING: benchmark-index refresh failed: {type(exc).__name__}: {exc}")

    print("Training complete!")
    print(f"Best validation positive Dice: {best_val_dice:.6f} (epoch {best_epoch})")
    print(f"Final test Dice (all): {test_results['all']['dice']:.6f}")
    print(f"Final test IoU (all): {test_results['all']['iou']:.6f}")
    print(f"Report: {experiment_paths.root / config.REPORT_FILE}")
    print(f"Audit: {experiment_paths.root / 'audit'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Phase 1: if a run has already been created, persist the failure
        # instead of leaving the run with a false "running" state.
        try:
            run_args = parse_args()
            if run_args.run_id:
                manager = ExperimentManager()
                paths = manager.get_paths(run_args.run_id)
                manifest = manager.load_manifest(paths)
                stage = manifest.get("stage") or "TRAINING"
                manager.mark_failed(paths, f"{type(exc).__name__}: {exc}", stage=stage)
        except Exception as state_error:
            print(f"WARNING: could not persist failed run state: {state_error}")
        traceback.print_exc()
        raise