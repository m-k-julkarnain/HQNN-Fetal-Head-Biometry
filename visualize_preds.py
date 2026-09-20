""" Inference & Visualization Script for Fetal Head Biometry HQNN. """
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.hqnn_model import FetalHeadHQNN
from src.datasets.head_dataset import FetalHeadDataset
from configs.paths import HEAD_FP_TEST_CSV, HEAD_FP_IMAGES
from configs.experiment import IMAGE_SIZE, NUM_QUBITS, QUANTUM_LAYERS

def run_visualization():
    print("============================================================")
    print("Generating Fetal Head Dual-Diameter Visualizations")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)

    checkpoint_path = "outputs/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}. Train the model first.")
        
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    val_dataset = FetalHeadDataset(csv_file=HEAD_FP_TEST_CSV, image_dir=HEAD_FP_IMAGES, image_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    images, true_coords, image_ids = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        pred_coords = model(images)
        if isinstance(pred_coords, tuple): 
            pred_coords = pred_coords[0]
        pred_coords = pred_coords.cpu()

    # Denormalize coordinates back to pixel space (224x224)
    true_coords = ((true_coords.cpu() + 1.0) / 2.0) * IMAGE_SIZE
    pred_coords = ((pred_coords + 1.0) / 2.0) * IMAGE_SIZE
    images = images.cpu()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Fetal Head HQNN: BPD & OFD Biometry Predictions", fontsize=14, y=0.98, fontweight='bold')
    
    # ImageNet reverse normalization for visual display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].text(0.5, -0.06, f"Image ID: {image_ids[i]}", ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        
        # Plot BPD (Green = True, Red = Predicted)
        tx1_b, ty1_b, tx2_b, ty2_b = true_coords[i, :4]
        px1_b, py1_b, px2_b, py2_b = pred_coords[i, :4]
        axes[i].plot([tx1_b, tx2_b], [ty1_b, ty2_b], 'go-', label="True BPD", markersize=6, linewidth=2)
        axes[i].plot([px1_b, px2_b], [py1_b, py2_b], 'ro-', label="Pred BPD", markersize=6, linewidth=2)
        
        # Plot OFD (Cyan = True, Magenta = Predicted)
        tx1_o, ty1_o, tx2_o, ty2_o = true_coords[i, 4:]
        px1_o, py1_o, px2_o, py2_o = pred_coords[i, 4:]
        axes[i].plot([tx1_o, tx2_o], [ty1_o, ty2_o], 'co-', label="True OFD", markersize=6, linewidth=2)
        axes[i].plot([px1_o, px2_o], [py1_o, py2_o], 'magenta', marker='o', linestyle='-', label="Pred OFD", markersize=6, linewidth=2)
        
        axes[i].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=4, fontsize=10)
    
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/bpd_ofd_predictions.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nSuccess! Visualization saved locally to: {save_path}")

if __name__ == "__main__":
    run_visualization()