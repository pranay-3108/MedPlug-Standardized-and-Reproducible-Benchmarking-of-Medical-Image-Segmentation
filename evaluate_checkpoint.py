import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data_loader import SIIMDataset
from eval import evaluate
from core.plugin_loader import load_model_plugin
from research_trace import extract_model_state


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIRECTORY / f"{config.CHECKPOINT_NAME}.pth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=config.DEVICE)
    parser.add_argument("--model", choices=["unet", "attention_unet", "transunet", "segformer"], default=config.MODEL)
    parser.add_argument("--pretrained", action="store_true", default=config.PRETRAINED)
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=config.INPUT_CHANNELS)
    parser.add_argument("--output-channels", type=int, default=config.OUTPUT_CHANNELS)
    parser.add_argument("--normalization", choices=["minmax", "imagenet"], default=config.NORMALIZATION)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    csv_map = {"train": config.TRAIN_CSV, "val": config.VAL_CSV, "test": config.TEST_CSV}
    dicom_map = {"train": config.DICOM_TRAIN_DIRECTORY, "val": config.DICOM_TRAIN_DIRECTORY,
                 "test": config.DICOM_TRAIN_DIRECTORY}
    csv_path = csv_map[args.split]
    dataset = SIIMDataset(csv_path, dicom_map[args.split], args.input_channels, args.normalization, config.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    model = load_model_plugin(args.model, in_channels=args.input_channels, out_channels=args.output_channels, pretrained=args.pretrained).to(device)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(extract_model_state(loaded), strict=True)

    results = evaluate(model, loader, device, max_batches=args.max_batches,
                       threshold=config.SEGMENTATION_THRESHOLD,
                       image_level_fraction=config.IMAGE_LEVEL_MIN_MASK_FRACTION)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Images: {len(dataset)}")
    print(f"Dice (all): {results['all']['dice']:.6f}")
    print(f"IoU (all): {results['all']['iou']:.6f}")
    print(f"Dice (positive): {results['positive']['dice']:.6f}")
    print(f"IoU (positive): {results['positive']['iou']:.6f}")
    print(f"HD95 (positive): {results['positive'].get('hd95', float('nan'))}")
    print(f"Precision: {results['all']['precision']:.6f}")
    print(f"Recall: {results['all']['recall']:.6f}")
    print(f"Specificity: {results['all']['specificity']:.6f}")
    print(f"MCC: {results['all']['mcc']:.6f}")
    print(f"Image F1: {results['image_classification']['f1']:.6f}")
    print(f"Image ROC-AUC: {results['image_classification']['roc_auc']}")
    print(f"Image PR-AUC: {results['image_classification']['pr_auc']}")


if __name__ == "__main__":
    main()
