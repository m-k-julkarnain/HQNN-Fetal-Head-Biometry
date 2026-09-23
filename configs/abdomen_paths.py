from pathlib import Path
from configs.paths import ROOT_DIR, ANNOTATIONS_DIR, IMAGES_DIR

# Abdomen Specific Directories
ABDOMEN_IMAGES_DIR = IMAGES_DIR / "Abdomen"

# Abdomen CSV Paths (FP, UCL, Multicentre - HC18 excluded)
ABDOMEN_FP_TRAIN_CSV = ANNOTATIONS_DIR / "FP" / "Abdomen_Train.csv"
ABDOMEN_FP_TEST_CSV = ANNOTATIONS_DIR / "FP" / "Abdomen_Test.csv"

ABDOMEN_UCL_CSV = ANNOTATIONS_DIR / "UCL" / "Abdomen.csv"

ABDOMEN_MULTICENTRE_TRAIN_CSV = ANNOTATIONS_DIR / "MULTICENTRE" / "Abdomen_Train.csv"
ABDOMEN_MULTICENTRE_TEST_CSV = ANNOTATIONS_DIR / "MULTICENTRE" / "Abdomen_Test.csv"