""" Main training script for Fetal Head Biometry HQNN. """
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.experiment import (
    EXPERIMENT_NAME, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_QUBITS,
    QUANTUM_LAYERS, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOPPING_PATIENCE, MIN_DELTA, USE_AMP, SEED
)
from configs.paths import HEAD_FP_TRAIN_CSV, HEAD_FP_TEST_CSV, HEAD_FP_IMAGES
from src.datasets.head_dataset import FetalHeadDataset
from src.models.hqnn_model import FetalHeadHQNN
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.train_one_epoch import train_one_epoch
from src.training.validate import validate
from src.training.checkpoint import save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_hqnn():
    print(f"============================================================")
    print(f"Starting Training Experiment: {EXPERIMENT_NAME}")
    print(f"============================================================")
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    train_dataset = FetalHeadDataset(csv_file=HEAD_FP_TRAIN_CSV, image_dir=HEAD_FP_IMAGES, image_size=IMAGE_SIZE)
    val_dataset = FetalHeadDataset(csv_file=HEAD_FP_TEST_CSV, image_dir=HEAD_FP_IMAGES, image_size=IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    
    model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)
    criterion = build_loss()
    optimizer = build_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP) if device.type == 'cuda' else None
    
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n------------------------------------------------------------")
        print(f"Epoch {epoch}/{EPOCHS}")
        
        train_loss = train_one_epoch(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device)
        val_loss, predictions, metrics = validate(model=model, dataloader=val_loader, criterion=criterion, device=device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss : {train_loss:.4f} | Val Loss : {val_loss:.4f} | Mean MAE : {metrics['mean_mae']:.4f} px")
        
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics, "learning_rate": current_lr}
        history.append(epoch_record)
        
        is_best = val_loss < (best_val_loss - MIN_DELTA)
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, history=history, metrics=metrics, best_loss=best_val_loss, is_best=is_best)
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered. No validation improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

if __name__ == "__main__":
    train_hqnn()