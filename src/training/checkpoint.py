""" Checkpoint manager for saving model states locally. """
from pathlib import Path
import torch

CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, metrics, best_loss, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "metrics": metrics,
        "best_loss": best_loss,
    }
    
    last_path = CHECKPOINT_DIR / "last_model.pth"
    torch.save(checkpoint, last_path)
    
    if is_best:
        best_path = CHECKPOINT_DIR / "best_model.pth"
        torch.save(checkpoint, best_path)
        print("Checkpoint saved: best_model.pth (New Best Validation Score)")