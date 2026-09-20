""" 25-Image Grid Inference & Visualization Script for Fetal Head Biometry. """
import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from huggingface_hub import HfApi

from src.models.hqnn_model import FetalHeadHQNN
from src.datasets.head_dataset import FetalHeadDataset
from configs.paths import HEAD_FP_TEST_CSV, HEAD_FP_IMAGES
from configs.experiment import IMAGE_SIZE, NUM_QUBITS, QUANTUM_LAYERS

def run_visualization():
    print("============================================================")
    print("Generating 25-Image Dual-Diameter Grid Visualization")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)

    checkpoint_path = "outputs/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    # Load exactly 25 images
    val_dataset = FetalHeadDataset(csv_file=HEAD_FP_TEST_CSV, image_dir=HEAD_FP_IMAGES, image_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=True)
    images, true_coords, image_ids = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        pred_coords = model(images)
        if isinstance(pred_coords, tuple): 
            pred_coords = pred_coords[0]
        pred_coords = pred_coords.cpu()

    true_coords = ((true_coords.cpu() + 1.0) / 2.0) * IMAGE_SIZE
    pred_coords = ((pred_coords + 1.0) / 2.0) * IMAGE_SIZE
    images = images.cpu()

    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    fig.suptitle("Fetal Head HQNN: BPD & OFD Biometry (25 Random Test Samples)", fontsize=24, y=0.92, fontweight='bold')
    axes = axes.flatten()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(25):
        img = images[i].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].text(0.5, -0.06, f"{image_ids[i]}", ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        
        tx1_b, ty1_b, tx2_b, ty2_b = true_coords[i, :4]
        px1_b, py1_b, px2_b, py2_b = pred_coords[i, :4]
        axes[i].plot([tx1_b, tx2_b], [ty1_b, ty2_b], 'go-', markersize=4, linewidth=1.5, label="True BPD" if i==0 else "")
        axes[i].plot([px1_b, px2_b], [py1_b, py2_b], 'ro-', markersize=4, linewidth=1.5, label="Pred BPD" if i==0 else "")
        
        tx1_o, ty1_o, tx2_o, ty2_o = true_coords[i, 4:]
        px1_o, py1_o, px2_o, py2_o = pred_coords[i, 4:]
        axes[i].plot([tx1_o, tx2_o], [ty1_o, ty2_o], 'co-', markersize=4, linewidth=1.5, label="True OFD" if i==0 else "")
        axes[i].plot([px1_o, px2_o], [py1_o, py2_o], 'magenta', marker='o', linestyle='-', markersize=4, linewidth=1.5, label="Pred OFD" if i==0 else "")
        
        axes[i].axis('off')

    fig.legend(loc='upper right', bbox_to_anchor=(0.90, 0.92), ncol=4, fontsize=16)
    
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/grid_25_predictions.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    
    print("\n>>> Pushing 25-image visualization grid to Hugging Face...")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        api = HfApi(token=token)
        repo_id = "m-k-julkarnain/HQNN-Fetal-Head-Biometry"
        
        api.upload_file(
            path_or_fileobj=save_path,
            path_in_repo="visualizations/grid_25_predictions.png",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Success! Grid uploaded to {repo_id}/visualizations/")
    except Exception as e:
        print(f"Hugging Face upload failed: {e}")

if __name__ == "__main__":
    run_visualization()