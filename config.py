from pathlib import Path


# ---------------------------
# Experiment Information
# ---------------------------
EXPERIMENT_NAME = "siim_acr_benchmark"
MODEL = "unet"
MODEL_NAME = MODEL
RANDOM_SEED = 42


# ---------------------------
# Model Configuration
# ---------------------------
PRETRAINED = False
INPUT_CHANNELS = 3  # grayscale image replicated to 3 channels
OUTPUT_CHANNELS = 1


# ---------------------------
# Data Configuration
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIRECTORY / "train-rle.csv"
VAL_CSV = DATA_DIRECTORY / "val-rle.csv"
TEST_CSV = DATA_DIRECTORY / "test-rle.csv"

DICOM_TRAIN_DIRECTORY = PROJECT_ROOT / "images" / "dicom-images-train"
DICOM_TEST_DIRECTORY = DICOM_TRAIN_DIRECTORY

IMAGE_SIZE = (256, 256)
NORMALIZATION = "minmax"
BATCH_SIZE = 2
NUM_WORKERS = 0


# ---------------------------
# Training Configuration
# ---------------------------
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adamw"
SCHEDULER = "cosine"
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = "auto"
RESUME_TRAINING = False

# Primary benchmark comparison uses a fixed epoch budget.
EARLY_STOPPING_ENABLED = False
PATIENCE = 10

# Probability threshold for segmentation masks.
SEGMENTATION_THRESHOLD = 0.50

# Fixed image-level disease decision derived from the predicted mask.
# 0.1% of pixels must be predicted positive to classify an image as positive.
IMAGE_LEVEL_MIN_MASK_FRACTION = 0.001

# Combined segmentation loss.
BCE_WEIGHT = 0.5


# ---------------------------
# Checkpoint Configuration
# ---------------------------
CHECKPOINT_DIRECTORY = PROJECT_ROOT / "checkpoints"
CHECKPOINT_NAME = f"{MODEL_NAME}_best_model"
RESUME_CHECKPOINT_NAME = f"{MODEL_NAME}_latest_training.pth"


# ---------------------------
# Logging Configuration
# ---------------------------
LOG_DIRECTORY = PROJECT_ROOT / "logs"


# ---------------------------
# Experiment Output Configuration
# ---------------------------
EXPERIMENTS_DIRECTORY = PROJECT_ROOT / "experiments"
TEMP_DIRECTORY = EXPERIMENTS_DIRECTORY / "temp"

BENCHMARK_RESULTS_FILE = "benchmark_results.csv"
LEADERBOARD_FILE = "leaderboard.csv"
MODEL_COMPARISON_FILE = "model_comparison.png"

CONFIG_FILE = "config.txt"
METRICS_FILE = "metrics.csv"
TRAINING_LOG_FILE = "training_log.csv"
REPORT_FILE = "report.txt"
REPORT_PDF_FILE = "report.pdf"
OVERLAY_FILE = "overlay.png"
PREDICTION_FILE = "prediction.png"
GROUND_TRUTH_FILE = "ground_truth.png"
TRAINING_LOSS_PLOT_FILE = "training_loss_curve.png"
VALIDATION_METRICS_PLOT_FILE = "validation_metrics_curve.png"
TEST_METRICS_PLOT_FILE = "test_metrics.png"

# Research audit artifacts. The regular run artifacts remain unchanged for compatibility.
AUDIT_DIR = "audit"
METRIC_LEDGER_FILE = "metric_ledger.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
VALIDATION_PREDICTIONS_FILE = "validation_predictions.csv"
VERIFICATION_FILE = "verification.json"
CHECKPOINT_HASHES_FILE = "checkpoint_hashes.json"
CHECKPOINT_METADATA_FILE = "best_checkpoint_metadata.json"
ARCHITECTURE_FILE = "architecture.txt"
REPRODUCIBILITY_GUIDE_FILE = "REPRODUCIBILITY_GUIDE.md"
RUN_COMMAND_FILE = "run_command.txt"
DATASET_STATS_FILE = "dataset_stats.json"
PREDICTION_BEHAVIOR_FILE = "prediction_behavior.png"
LEARNING_RATE_FILE = "learning_rate_curve.png"
PROBABILITY_FILE = "probability_stats.png"
LOGIT_FILE = "logit_stats.png"
GRADIENT_FILE = "gradient_stability.png"
CONFUSION_FILE = "test_confusion_matrix.png"
FORENSICS_FILE = "run_forensics.json"
