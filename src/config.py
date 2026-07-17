from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "GermEval-2018-Data-master"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PERTURBED_DIR = PROJECT_ROOT / "data" / "perturbed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILENAME = "germeval2018.training.txt"
TEST_FILENAME = "germeval2018.test.txt"

SEED = 42
VAL_SIZE = 0.15

PERTURBATION_TYPES = ["typo", "casing", "umlaut", "elongation", "slang"]
PERTURBATION_INTENSITIES = [0.05, 0.10, 0.20]

BASELINE_NGRAM_RANGES = [(1, 1), (1, 2)]
BASELINE_C_GRID = [0.01, 0.1, 1.0, 10.0]
