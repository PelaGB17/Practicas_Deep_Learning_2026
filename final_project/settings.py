from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "03TransferLearning" / "dataset" / "training"
DEFAULT_VAL_DIR = PROJECT_ROOT / "03TransferLearning" / "dataset" / "validation"

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "final_project" / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "final_project" / "reports"

DEFAULT_CHECKPOINT_PATH = DEFAULT_ARTIFACTS_DIR / "resnet50_scene_best.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
