""" Fully Rewritten Master LODO Ablation Pipeline for Fetal Abdomen Biometry. """
import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient
from configs.experiment import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_QUBITS,
    QUANTUM_LAYERS, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    EARLY_STOPPING_PATIENCE, MIN_DELTA, USE_AMP, SEED
)
from configs.abdomen_paths import *
from src.datasets.abdomen_dataset import FetalAbdomenDataset
from src.models.classical_baselines import PureDenseNet121
from src.models.hqnn_model import FetalHeadHQNN
from src.training.loss import build_loss
from src.training.optimizer import build_optimizer
from src.training.train_one_epoch import train_one_epoch
from src.training.validate import validate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_concat_dataset(config_list):
    datasets = []
    for csv_path, img_dir in config_list:
        datasets.append(FetalAbdomenDataset(csv_file=csv_path, image_dir=img_dir, image_size=IMAGE_SIZE))
    return ConcatDataset(datasets)

def train_and_evaluate(model, model_name, run_name, train_loader, val_loader, device, hf_api, repo_id):
    print(f"\n{'='*60}\nArchitecture: {model_name} | {run_name} [Optimized Loss & Sched]\n{'='*60}")
    criterion = build_loss()
    optimizer = build_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing Warm Restarts to escape plateau traps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP) if device.type == 'cuda' else None

    swa_model = AveragedModel(model)
    swa_start = int(EPOCHS * 0.70)
    swa_scheduler = SWALR(optimizer, swa_lr=LEARNING_RATE * 0.05)

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs("outputs/checkpoints", exist_ok=True)
    checkpoint_path = f"outputs/checkpoints/best_Abdomen_{run_name}_{model_name}.pth"

    best_detailed_records = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        scheduler.step()

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            val_model = swa_model
        else:
            val_model = model

        val_loss, detailed_records, metrics = validate(val_model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {metrics['mean_mae']:.4f} px {'[SWA Active]' if epoch >= swa_start else ''}")

        if val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(val_model.module.state_dict() if hasattr(val_model, 'module') else val_model.state_dict(), checkpoint_path)
            best_detailed_records = detailed_records
        else:
            patience_counter += 1
            if patience_counter >= (EARLY_STOPPING_PATIENCE + 10) and epoch < swa_start:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break

    if epoch >= swa_start:
        print(">>> Updating SWA Batch Norm Statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    os.makedirs("outputs/results", exist_ok=True)
    csv_path = f"outputs/results/CDF_Data_Abdomen_{run_name}_{model_name}.csv"
    df = pd.DataFrame(best_detailed_records)
    df["Model_Architecture"] = model_name
    df["LODO_Run"] = run_name
    df.to_csv(csv_path, index=False)
    print(f"\n>>> CDF Dataset saved: {csv_path}")

    if hf_api:
        hf_api.upload_file(path_or_fileobj=checkpoint_path, path_in_repo=f"checkpoints/Abdomen_{run_name}_{model_name}.pth", repo_id=repo_id)
        hf_api.upload_file(path_or_fileobj=csv_path, path_in_repo=f"results/CDF_Abdomen_{run_name}_{model_name}.csv", repo_id=repo_id)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    generate_25_image_grid(model, model_name, run_name, val_loader, device, hf_api, repo_id)

def generate_25_image_grid(model, model_name, run_name, val_loader, device, hf_api, repo_id):
    print(f">>> Generating Abdomen Publication Grid for {model_name}...")
    model.eval()
    batch_images, batch_coords, batch_ids = [], [], []
    for imgs, coords, ids in val_loader:
        batch_images.append(imgs)
        batch_coords.append(coords)
        batch_ids.extend(ids)
        if sum(b.size(0) for b in batch_images) >= 25: break

    images = torch.cat(batch_images, dim=0)[:25].to(device)
    true_coords = torch.cat(batch_coords, dim=0)[:25].to(device)
    image_ids = batch_ids[:25]

    with torch.no_grad():
        pred_coords = model(images)
        if isinstance(pred_coords, tuple): pred_coords = pred_coords[0]
    true_coords = ((true_coords.cpu() + 1.0) / 2.0) * IMAGE_SIZE
    pred_coords = ((pred_coords.cpu() + 1.0) / 2.0) * IMAGE_SIZE
    images = images.cpu()

    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    fig.suptitle(f"Abdomen | {run_name} | {model_name} | TAD & APAD Biometry", fontsize=24, y=0.92, fontweight='bold')
    axes = axes.flatten()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    for i in range(25):
        img = np.clip(std * images[i].permute(1, 2, 0).numpy() + mean, 0, 1)
        axes[i].imshow(img)
        axes[i].text(0.5, -0.06, f"{image_ids[i]}", ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        
        # TAD
        tx1_b, ty1_b, tx2_b, ty2_b = true_coords[i, :4]
        px1_b, py1_b, px2_b, py2_b = pred_coords[i, :4]
        axes[i].plot([tx1_b, tx2_b], [ty1_b, ty2_b], 'go-', markersize=4, linewidth=1.5, label="True TAD" if i==0 else "")
        axes[i].plot([px1_b, px2_b], [py1_b, py2_b], 'ro-', markersize=4, linewidth=1.5, label="Pred TAD" if i==0 else "")
        
        # APAD
        tx1_o, ty1_o, tx2_o, ty2_o = true_coords[i, 4:]
        px1_o, py1_o, px2_o, py2_o = pred_coords[i, 4:]
        axes[i].plot([tx1_o, tx2_o], [ty1_o, ty2_o], 'co-', markersize=4, linewidth=1.5, label="True APAD" if i==0 else "")
        axes[i].plot([px1_o, px2_o], [py1_o, py2_o], 'magenta', marker='o', linestyle='-', markersize=4, linewidth=1.5, label="Pred APAD" if i==0 else "")
        axes[i].axis('off')

    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.92), ncol=4, fontsize=16)
    os.makedirs("outputs/figures", exist_ok=True)
    grid_path = f"outputs/figures/Grid_25_Abdomen_{run_name}_{model_name}.png"
    plt.savefig(grid_path, bbox_inches='tight', dpi=200)
    plt.close()

    if hf_api:
        hf_api.upload_file(path_or_fileobj=grid_path, path_in_repo=f"visualizations/Grid_Abdomen_{run_name}_{model_name}.png", repo_id=repo_id)

def main():
    parser = argparse.ArgumentParser(description="Abdomen LODO Benchmark")
    parser.add_argument("--run", type=int, required=True, choices=[1, 2, 3], help="Abdomen LODO Run Number (1-3)")
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LODO_CONFIGS = {
        1: {"name": "Run1_Target_UCL", "train": [(ABDOMEN_FP_TRAIN_CSV, ABDOMEN_IMAGES_DIR)], "test": (ABDOMEN_UCL_CSV, ABDOMEN_IMAGES_DIR)},
        2: {"name": "Run2_Target_FP", "train": [(ABDOMEN_UCL_CSV, ABDOMEN_IMAGES_DIR)], "test": (ABDOMEN_FP_TEST_CSV, ABDOMEN_IMAGES_DIR)},
        3: {"name": "Run3_Target_MULTICENTRE", "train": [(ABDOMEN_MULTICENTRE_TRAIN_CSV, ABDOMEN_IMAGES_DIR)], "test": (ABDOMEN_MULTICENTRE_TEST_CSV, ABDOMEN_IMAGES_DIR)}
    }
    config = LODO_CONFIGS[args.run]
    run_name = config["name"]
    print(f"\n{'*'*60}\nInitiating Abdomen {run_name}\n{'*'*60}")

    train_dataset = create_concat_dataset(config["train"])
    test_dataset = FetalAbdomenDataset(csv_file=config["test"][0], image_dir=config["test"][1], image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
        hf_api = HfApi(token=token)
        repo_id = "m-k-julkarnain/HQNN-Fetal-Head-Biometry"
        hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
        print("Hugging Face API connected.")
    except:
        hf_api, repo_id = None, None

    classical_model = PureDenseNet121().to(device)
    train_and_evaluate(classical_model, "Classical_DenseNet121", run_name, train_loader, test_loader, device, hf_api, repo_id)

    quantum_model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)
    train_and_evaluate(quantum_model, "Quantum_HQNN", run_name, train_loader, test_loader, device, hf_api, repo_id)

if __name__ == "__main__":
    main()