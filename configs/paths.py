from pathlib import Path

# Automatically detect environment
IS_KAGGLE = Path('/kaggle/input').exists()

if IS_KAGGLE:
    # Dynamic search for Kaggle dataset directory
    possible_paths = list(Path('/kaggle/input').rglob('FetalBiometry-Multicentre-Landmarks-2026'))
    if possible_paths:
        ROOT_DIR = possible_paths[0]
    else:
        ROOT_DIR = Path('/kaggle/input/datasets/mkjulkarnain/multicentre-fetal-biometry-2026/FetalBiometry-Multicentre-Landmarks-2026')
else:
    # Local Mac/PC dataset path
    ROOT_DIR = Path("data/raw/FetalBiometry-Multicentre-Landmarks-2026")

# Core Subdirectory Layout
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
IMAGES_DIR = ROOT_DIR / "images"

# Head-Specific Paths (FP Split Baseline)
HEAD_FP_TRAIN_CSV = ANNOTATIONS_DIR / "FP" / "Head_Train.csv"
HEAD_FP_TEST_CSV = ANNOTATIONS_DIR / "FP" / "Head_Test.csv"
HEAD_FP_IMAGES = IMAGES_DIR / "FP" / "Head"

# Multi-Centre & UCL Validation Paths
HEAD_MULTICENTRE_CSV = ANNOTATIONS_DIR / "MULTICENTRE" / "Head.csv"
HEAD_MULTICENTRE_IMAGES = IMAGES_DIR / "MULTICENTRE" / "Head"
HEAD_UCL_CSV = ANNOTATIONS_DIR / "UCL" / "Head.csv"
HEAD_UCL_IMAGES = IMAGES_DIR / "UCL" / "Head"
HEAD_UCL_PX_TO_MM = ANNOTATIONS_DIR / "UCL" / "px_to_mm" / "Head.csv"