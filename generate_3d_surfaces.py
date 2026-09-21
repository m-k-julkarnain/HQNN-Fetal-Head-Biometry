""" 3D Attribution Relief Surface Generator for HQNN. """
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from huggingface_hub import HfApi

from src.models.hqnn_model import FetalHeadHQNN
from src.datasets.head_dataset import FetalHeadDataset
from configs.experiment import IMAGE_SIZE, NUM_QUBITS, QUANTUM_LAYERS
from configs.paths import HEAD_UCL_CSV, HEAD_UCL_IMAGES

def generate_3d_surface():
    print("============================================================")
    print("Generating 3D Anatomical Confidence Relief Surface")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)
    
    checkpoint_path = "outputs/checkpoints/best_Run1_Target_UCL_Quantum_HQNN.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()

    dataset = FetalHeadDataset(csv_file=HEAD_UCL_CSV, image_dir=HEAD_UCL_IMAGES, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    img, _, image_id = next(iter(loader))
    img = img.to(device)

    with torch.no_grad():
        _, mask_logits = model(img)
        # Apply sigmoid to get raw probability density [0, 1]
        prob_map = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

    # Generate the 3D Meshgrid
    X, Y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))

    # Matplotlib 3D Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface topography using Inferno colormap
    surf = ax.plot_surface(X, Y, prob_map, cmap=cm.inferno, linewidth=0, antialiased=True, alpha=0.9)
    
    # Add a floor contour map just like the Alzheimer's reference
    cset = ax.contourf(X, Y, prob_map, zdir='z', offset=-0.2, cmap=cm.Greys, alpha=0.5)

    ax.set_zlim(-0.2, 1.0)
    ax.set_title(f"Quantum HQNN 3D Anatomical Confidence\nUnseen UCL Target: {image_id[0]}", fontsize=16, fontweight='bold')
    ax.set_xlabel('Image X (px)')
    ax.set_ylabel('Image Y (px)')
    ax.set_zlabel('Confidence P(x,y)')
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label="Attribution Intensity")

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = f"outputs/figures/3D_Relief_UCL_Quantum_{image_id[0].replace('.png', '')}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f">>> 3D Plot saved locally to {save_path}")

    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        api = HfApi(token=token)
        repo_id = "m-k-julkarnain/HQNN-Fetal-Head-Biometry"
        api.upload_file(path_or_fileobj=save_path, path_in_repo=f"visualizations/3D_Relief_UCL_Quantum.png", repo_id=repo_id)
        print(f">>> Uploaded 3D Relief to Hugging Face!")
    except Exception as e:
        print("HF Upload Skipped.")

if __name__ == "__main__":
    generate_3d_surface()