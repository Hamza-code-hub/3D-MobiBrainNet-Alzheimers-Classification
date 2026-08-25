from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

CLASS_NAMES = [
    "AD",
    "MCI",
    "CN",
]

VOLUME_SIZE = (
    96,
    96,
    96,
)

NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 30

RANDOM_SEED = 42
