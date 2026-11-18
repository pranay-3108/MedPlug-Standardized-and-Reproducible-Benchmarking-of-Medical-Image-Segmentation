import hashlib
import math

import numpy as np
import torch


def _threshold_logits(logits, threshold):
    probabilities = torch.sigmoid(logits)
    return (probabilities > threshold).float(), probabilities


def _binary_counts(pred, target):
    pred = pred.bool()
    target = target.bool()
    tp = torch.logical_and(pred, target).sum().item()
    tn = torch.logical_and(~pred, ~target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()
    return tp, tn, fp, fn


def dice_score(pred, target, eps=1e-6, threshold=0.5):
    pred_binary, _ = _threshold_logits(pred, threshold)
    target = target.float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def iou_score(pred, target, eps=1e-6, threshold=0.5):
    pred_binary, _ = _threshold_logits(pred, threshold)
    target = target.float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def _mcc(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def _boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int32)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[1:-1, 1:-1]
    )
    edge = mask & ~eroded
    return np.argwhere(edge).astype(np.int32)


def _directed_boundary_distances(a, b, chunk_size=256):
    if len(a) == 0 or len(b) == 0:
        return np.empty((0,), dtype=np.float32)
    distances = []
    b_float = b.astype(np.float32)
    for start in range(0, len(a), chunk_size):
        chunk = a[start : start + chunk_size].astype(np.float32)
        delta = chunk[:, None, :] - b_float[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def hd95_single(pred_mask, target_mask):
    """95th-percentile symmetric boundary distance for one positive case.

    For a positive ground-truth case with an empty prediction, the image diagonal
    is used as a fixed miss penalty. True-negative images are reported separately.
    """
    target_boundary = _boundary(target_mask)
    pred_boundary = _boundary(pred_mask)

    if len(target_boundary) == 0 and len(pred_boundary) == 0:
        return float("nan")

    if len(target_boundary) == 0 or len(pred_boundary) == 0:
        h, w = target_mask.shape
        return float(np.hypot(h - 1, w - 1))

    d1 = _directed_boundary_distances(pred_boundary, target_boundary)
    d2 = _directed_boundary_distances(target_boundary, pred_boundary)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def _roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_total = int(labels.sum())
    if pos_total == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos_total
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall)) if hasattr(np, "trapezoid") else float(np.trapz(precision, recall))


def evaluate(model, loader, device=None, max_batches=None, threshold=0.5, image_level_fraction=0.001, collect_cases=False, collect_visual_samples=False):
    """Evaluate segmentation and image-level disease prediction.

    When collect_visual_samples is enabled, compact pixel arrays are returned
    from the same inference pass so later research figures do not need to
    rerun the model on test data.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    groups = {
        "all": {"dice": [], "iou": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "positive": {"dice": [], "iou": [], "hd95": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "negative": {"dice": [], "iou": [], "tp": 0, "tn": 0, "fp": 0, "fn": 0},
    }
    image_labels = []
    image_scores = []
    image_predictions = []
    case_records = []
    visual_samples = []
    case_index = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            pred_binary, probabilities = _threshold_logits(outputs, threshold)
            target = masks.float()

            intersection = (pred_binary * target).sum(dim=(1, 2, 3))
            pred_area = pred_binary.sum(dim=(1, 2, 3))
            target_area = target.sum(dim=(1, 2, 3))
            union = pred_area + target_area - intersection
            dice = (2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)
            iou = (intersection + 1e-6) / (union + 1e-6)
            both_empty = (pred_area == 0) & (target_area == 0)
            dice = torch.where(both_empty, torch.ones_like(dice), dice)
            iou = torch.where(both_empty, torch.ones_like(iou), iou)
            positive_target = target_area > 0

            for i in range(images.shape[0]):
                pred_np = pred_binary[i, 0].detach().cpu().numpy().astype(bool)
                target_np = target[i, 0].detach().cpu().numpy().astype(bool)
                group = "positive" if positive_target[i].item() else "negative"
                groups["all"]["dice"].append(float(dice[i].item()))
                groups["all"]["iou"].append(float(iou[i].item()))
                groups[group]["dice"].append(float(dice[i].item()))
                groups[group]["iou"].append(float(iou[i].item()))

                if group == "positive":
                    groups[group]["hd95"].append(hd95_single(pred_np, target_np))

                tp, tn, fp, fn = _binary_counts(pred_binary[i], target[i])
                for key, value in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
                    groups["all"][key] += value
                    groups[group][key] += value

                score = float(probabilities[i].mean().item())
                image_label = int(positive_target[i].item())
                image_prediction = int((pred_area[i].item() / target[i].numel()) >= image_level_fraction)
                image_labels.append(image_label)
                image_scores.append(score)
                image_predictions.append(image_prediction)
                image_id = None
                if collect_cases or collect_visual_samples:
                    if hasattr(loader.dataset, "ids") and case_index < len(loader.dataset.ids):
                        image_id = str(loader.dataset.ids[case_index])
                    else:
                        image_id = f"case_{case_index:06d}"

                if collect_cases:
                    prob_np = probabilities[i, 0].detach().cpu()
                    logit_np = outputs[i, 0].detach().cpu()
                    case_records.append({
                        "SampleIndex": case_index,
                        "ImageId": image_id,
                        "GroundTruthPositive": image_label,
                        "PredictionPositive": image_prediction,
                        "GTPositivePixels": int(target_area[i].item()),
                        "PredPositivePixels": int(pred_area[i].item()),
                        "GTPositiveFraction": float(target_area[i].item() / target[i].numel()),
                        "PredPositiveFraction": float(pred_area[i].item() / target[i].numel()),
                        "MeanProbability": float(prob_np.mean().item()),
                        "MaxProbability": float(prob_np.max().item()),
                        "MinProbability": float(prob_np.min().item()),
                        "LogitMean": float(logit_np.mean().item()),
                        "LogitMax": float(logit_np.max().item()),
                        "LogitMin": float(logit_np.min().item()),
                        "Dice": float(dice[i].item()),
                        "IoU": float(iou[i].item()),
                        "HD95": float(hd95_single(pred_np, target_np)) if group == "positive" else float("nan"),
                        "TP": int(tp),
                        "TN": int(tn),
                        "FP": int(fp),
                        "FN": int(fn),
                        "ImageScore": score,
                        "PredictionMaskHash": hashlib.sha256(pred_binary[i, 0].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                        "ProbabilityMapHash": hashlib.sha256(prob_np.numpy().tobytes()).hexdigest(),
                    })

                if collect_visual_samples:
                    image_np = images[i, 0].detach().float().cpu().numpy()
                    image_min = float(np.nanmin(image_np)) if image_np.size else 0.0
                    image_max = float(np.nanmax(image_np)) if image_np.size else 0.0
                    if not np.isfinite(image_min) or not np.isfinite(image_max) or image_max <= image_min:
                        image_u8 = np.zeros(image_np.shape, dtype=np.uint8)
                    else:
                        image_u8 = np.clip((image_np - image_min) * (255.0 / (image_max - image_min)), 0, 255).astype(np.uint8)
                    visual_samples.append({
                        "image_id": image_id,
                        "dice": float(dice[i].item()),
                        "image": image_u8,
                        "ground_truth": target_np.astype(np.uint8),
                        "prediction": pred_np.astype(np.uint8),
                    })
                case_index += 1

            if max_batches is not None and batch_idx >= max_batches:
                break

    def summarize(group):
        tp, tn, fp, fn = group["tp"], group["tn"], group["fp"], group["fn"]
        dice_mean = float(np.mean(group["dice"])) if group["dice"] else 0.0
        iou_mean = float(np.mean(group["iou"])) if group["iou"] else 0.0
        return {
            "dice": dice_mean,
            "iou": iou_mean,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "specificity": _safe_div(tn, tn + fp),
            "mcc": _mcc(tp, tn, fp, fn),
            "pixel_accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
            "fpr": _safe_div(fp, fp + tn),
            "fnr": _safe_div(fn, fn + tp),
            "count": len(group["dice"]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    image_labels_np = np.asarray(image_labels, dtype=np.int64)
    image_scores_np = np.asarray(image_scores, dtype=np.float64)
    image_predictions_np = np.asarray(image_predictions, dtype=np.int64)
    img_tp = int(((image_predictions_np == 1) & (image_labels_np == 1)).sum())
    img_tn = int(((image_predictions_np == 0) & (image_labels_np == 0)).sum())
    img_fp = int(((image_predictions_np == 1) & (image_labels_np == 0)).sum())
    img_fn = int(((image_predictions_np == 0) & (image_labels_np == 1)).sum())
    img_precision = _safe_div(img_tp, img_tp + img_fp)
    img_recall = _safe_div(img_tp, img_tp + img_fn)
    img_f1 = _safe_div(2 * img_precision * img_recall, img_precision + img_recall)
    image_classification = {
        "accuracy": _safe_div(img_tp + img_tn, len(image_labels_np)),
        "precision": img_precision,
        "recall": img_recall,
        "specificity": _safe_div(img_tn, img_tn + img_fp),
        "f1": img_f1,
        "roc_auc": _roc_auc(image_labels_np, image_scores_np),
        "pr_auc": _pr_auc(image_labels_np, image_scores_np),
        "fpr": _safe_div(img_fp, img_fp + img_tn),
        "fnr": _safe_div(img_fn, img_fn + img_tp),
        "tp": img_tp,
        "tn": img_tn,
        "fp": img_fp,
        "fn": img_fn,
        "count": len(image_labels_np),
    }

    results = {key: summarize(value) for key, value in groups.items()}
    results["positive"]["hd95"] = float(np.nanmean(groups["positive"]["hd95"])) if groups["positive"]["hd95"] else float("nan")
    results["image_classification"] = image_classification
    results["diagnostics"] = {
        "prediction_count": len(image_labels_np),
        "empty_prediction_rate": float(sum(1 for value in image_predictions_np if value == 0) / len(image_predictions_np)) if len(image_predictions_np) else 0.0,
        "pred_positive_fraction": float(np.mean([float(r.get("PredPositiveFraction", 0.0)) for r in case_records])) if case_records else float(np.mean([0.0])) if not image_labels_np.size else float("nan"),
        "target_positive_fraction": float(np.mean([float(r.get("GTPositiveFraction", 0.0)) for r in case_records])) if case_records else float("nan"),
        "mean_probability": float(np.mean([float(r.get("MeanProbability", 0.0)) for r in case_records])) if case_records else float("nan"),
        "max_probability": float(np.max([float(r.get("MaxProbability", 0.0)) for r in case_records])) if case_records else float("nan"),
    }
    if collect_cases:
        results["cases"] = case_records
    if collect_visual_samples:
        results["visual_samples"] = visual_samples
    return results
