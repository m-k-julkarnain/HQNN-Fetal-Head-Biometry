# Experiment Metadata
EXPERIMENT_NAME = "HQNN_Fetal_Head_Baseline"
MODEL_NAME = "FetalHeadHQNN"
DATASET_NAME = "Multicentre_Fetal_Biometry_2026"

# Dataset & Spatial Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Set to 0 to prevent multiprocessing background hangs

# Quantum Circuit Parameters
NUM_QUBITS = 8          # Doubled for exponential state space capacity
QUANTUM_LAYERS = 4      # Deepened for highly complex entanglement

# Optimization Parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001

# Mixed Precision & Reproducibility
USE_AMP = True
SEED = 42