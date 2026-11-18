import numpy as np
import torch


def decode_rle(encoded_pixels, shape, foreground=1):
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)

    encoded_pixels = str(encoded_pixels).strip()
    if not encoded_pixels or encoded_pixels == "-1":
        return mask.reshape((height, width), order="F")

    values = np.asarray(encoded_pixels.split(), dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]

    for start, length in zip(starts, lengths):
        end = start + length
        mask[start:end] = foreground

    return mask.reshape((height, width), order="F")


def encode_rle(mask):
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}.")

    mask = (mask > 0).astype(np.uint8)
    flat = mask.reshape(-1, order="F")
    flat = np.pad(flat, (1, 1), mode="constant")
    changes = np.where(flat[1:] != flat[:-1])[0] + 1

    runs = []
    for start, end in zip(changes[0::2], changes[1::2]):
        runs.extend([str(start), str(end - start)])

    return " ".join(runs)


def build_mask_from_rles(rles, shape):
    mask = np.zeros(shape, dtype=np.uint8)

    for encoded_pixels in rles:
        encoded_pixels = str(encoded_pixels).strip()
        if encoded_pixels in {"", "-1", "nan"}:
            continue

        mask |= decode_rle(encoded_pixels, shape)

    return mask


def rle2mask(rle, width, height):
    return decode_rle(rle, (height, width), foreground=255)


def mask2rle(img, width, height):
    if img.shape != (height, width):
        raise ValueError(f"Expected {(height, width)}, got {img.shape}.")

    return encode_rle(img)


class DiceBCELoss(torch.nn.Module):
    """Combined Dice loss and BCE loss to address class imbalance.
    
    This loss function combines the stability of BCE with the spatial
    awareness of Dice loss, which is particularly important for
    highly imbalanced segmentation tasks like pneumothorax detection
    where ~80% of images have no pathology.
    """
    def __init__(self, bce_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.eps) / (union + self.eps)).mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
