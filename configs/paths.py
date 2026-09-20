from pathlib import Path

IS_KAGGLE = Path('/kaggle/input').exists()

if IS_KAGGLE:
    possible_paths = list(Path('/kaggle/input').rglob('FetalBiometry-Multicentre-Landmarks-2026'))
    if possible_paths:
        ROOT_DIR = possible_paths[0]
    else:
        ROOT_DIR = Path('/kaggle/input/datasets/mkjulkarnain/multicentre-fetal-biometry-2026/FetalBiometry-Multicentre-Landmarks-2026')
else:
    ROOT_DIR = Path("data/raw/FetalBiometry-Multicentre-Landmarks-2026")

ANNOTATIONS_DIR = ROOT_DIR / "annotations"
IMAGES_DIR = ROOT_DIR / "images"

# Internal Domain
HEAD_FP_TRAIN_CSV = ANNOTATIONS_DIR / "FP" / "Head_Train.csv"
HEAD_FP_TEST_CSV = ANNOTATIONS_DIR / "FP" / "Head_Test.csv"
HEAD_FP_IMAGES = IMAGES_DIR / "FP" / "Head"

# External Domains (HC18 & UCL)
HEAD_HC18_CSV = ANNOTATIONS_DIR / "HC18" / "Head.csv"
HEAD_HC18_IMAGES = IMAGES_DIR / "HC18" / "Head"
HEAD_UCL_CSV = ANNOTATIONS_DIR / "UCL" / "Head.csv"
HEAD_UCL_IMAGES = IMAGES_DIR / "UCL" / "Head"

# Multicentre Upper Baseline Domain
HEAD_MULTICENTRE_TRAIN_CSV = ANNOTATIONS_DIR / "MULTICENTRE" / "Head_Train.csv"
HEAD_MULTICENTRE_TEST_CSV = ANNOTATIONS_DIR / "MULTICENTRE" / "Head_Test.csv"
HEAD_MULTICENTRE_IMAGES = IMAGES_DIR / "MULTICENTRE" / "Head"