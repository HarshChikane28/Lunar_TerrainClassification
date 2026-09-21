"""Project configuration and reproducible defaults."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ROOT_DIR / "Train_DATA"
TEST_DIR = ROOT_DIR / "Test_DATA"

TRAIN_METADATA = TRAIN_DIR / "train_metadata.csv"
TEST_METADATA = TEST_DIR / "test_metadata.csv"
TRAIN_IMAGE_DIR = TRAIN_DIR / "Train-20260918T131531Z-1-001" / "Train" / "train_images"
TEST_IMAGE_DIR = TEST_DIR / "Test-20260918T131433Z-1-001" / "Test" / "eval_images"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
SUBMISSION_PATH = ROOT_DIR / "submission.csv"

IMAGE_SIZE = 256
CANONICAL_AZIMUTH_DEGREES = 0.0
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20

BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 2
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
METADATA_DROPOUT = 0.10
CLASSIFIER_DROPOUT = 0.30
GRADIENT_ACCUMULATION_STEPS = 2
USE_AMP = True
USE_CHANNELS_LAST = True
COMPILE_MODEL = False

# A neutral grayscale normalization is appropriate for a single-channel image.
IMAGE_MEAN = (0.5,)
IMAGE_STD = (0.5,)
