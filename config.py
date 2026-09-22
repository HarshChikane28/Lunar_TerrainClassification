"""Project configuration and reproducible defaults."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ROOT_DIR / "Train_DATA"
TEST_DIR = ROOT_DIR / "Test_DATA"

TRAIN_METADATA = TRAIN_DIR / "train_metadata.csv"
TEST_METADATA = TEST_DIR / "test_metadata.csv"
TRAIN_IMAGE_DIR = TRAIN_DIR / "train_images"
TEST_IMAGE_DIR = TEST_DIR / "eval_images"
SUBMISSION_PATH = ROOT_DIR / "submission.csv"

IMAGE_SIZE = 256
CANONICAL_AZIMUTH_DEGREES = 0.0
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20

BATCH_SIZE = 16  # Conservative starting point for a 6 GB laptop GPU.
NUM_EPOCHS = 20
CHUNK_EPOCHS = 5
NUM_WORKERS = 2  # Windows + 16 GB RAM: enough overlap without excessive copies.
PREFETCH_FACTOR = 2
USE_AMP = True
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
METADATA_DROPOUT = 0.10
CLASSIFIER_DROPOUT = 0.30

# A neutral grayscale normalization is appropriate for a single-channel image.
IMAGE_MEAN = (0.5,)
IMAGE_STD = (0.5,)

RUN_DIR = ROOT_DIR / "runs" / "resnet18_fusion"
LATEST_CHECKPOINT = RUN_DIR / "latest.pt"
BEST_CHECKPOINT = RUN_DIR / "best_model.pt"
TRAINING_HISTORY = RUN_DIR / "training_history.csv"
VALIDATION_PREDICTIONS = RUN_DIR / "validation_predictions.csv"
BEST_VALIDATION_PREDICTIONS = RUN_DIR / "validation_predictions_best.csv"
DATA_AUDIT = RUN_DIR / "data_audit.csv"
RUN_SUMMARY = RUN_DIR / "run_summary.csv"
