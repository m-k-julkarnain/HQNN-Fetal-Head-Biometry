""" Master pipeline runner script for Fetal Head Biometry HQNN. """
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_hqnn

if __name__ == "__main__":
    print("============================================================")
    print("Starting Fetal Head HQNN Full Pipeline Execution")
    print("============================================================\n")
    
    print(">>> Phase 1: Model Training & Checkpoint Optimization")
    train_hqnn()
    
    print("\n============================================================")
    print("Pipeline Execution Completed Successfully.")
    print("============================================================")