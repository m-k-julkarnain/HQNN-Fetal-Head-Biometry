""" 3D Attribution Relief Surface Generator for HQNN (Parameterized for all Runs). """
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from huggingface_hub import HfApi

from src.models.hqnn_model import FetalHeadHQNN
from src.datasets.head_dataset import FetalHeadDataset
from configs.experiment import IMAGE_SIZE, NUM_QUBITS, QUANTUM_LAYERS
from configs.paths import *

def generate_3d_surface(run_num):
    # Dynamically map the run number to the exact target dataset and checkpoint
    LODO_TARGETS = {
        1: {"name": "Run1_Target_UCL", "csv": HEAD_UCL_CSV, "img_dir": HEAD_UCL_IMAGES, "title": "Unseen UCL Target"},
        2: {"name": "Run2_Target_HC18", "csv": HEAD_HC18_CSV, "img_dir": HEAD_HC18_IMAGES, "title": "Unseen HC18 Target"},
        3: {"name": "Run3_Target_FP", "csv": HEAD_FP_TEST_CSV, "img_dir": HEAD_FP_IMAGES, "title": "Unseen FP Target"},
        4: {"name": "Run4_Target_MULTICENTRE", "csv": HEAD_MULTICENTRE_TEST_CSV, "img_dir": HEAD_MULTICENTRE_IMAGES, "title": "Unseen MULTICENTRE Target"}
    }

    config = LODO_TARGETS[run_num]
    run_name = config["name"]
    
    print("============================================================")
    print(f"Generating 3D Anatomical Confidence Relief Surface for {run_name}")
    print("============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FetalHeadHQNN(n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS).to(device)
    
    checkpoint_path = f"outputs/checkpoints/best_{run_name}_Quantum_HQNN.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()

    dataset = FetalHeadDataset(csv_file=config["csv"], image_dir=config["img_dir"], image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    img, _, image_id = next(iter(loader))
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        # Handle tuple outputs safely
        mask_logits = outputs[1] if isinstance(outputs, tuple) else outputs
        prob_map = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

    X, Y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, prob_map, cmap=cm.inferno, linewidth=0, antialiased=True, alpha=0.9)
    cset = ax.contourf(X, Y, prob_map, zdir='z', offset=-0.2, cmap=cm.Greys, alpha=0.5)

    ax.set_zlim(-0.2, 1.0)
    ax.set_title(f"Quantum HQNN 3D Anatomical Confidence\n{config['title']}: {image_id[0]}", fontsize=16, fontweight='bold')
    ax.set_xlabel('Image X (px)')
    ax.set_ylabel('Image Y (px)')
    ax.set_zlabel('Confidence P(x,y)')
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label="Attribution Intensity")

    os.makedirs("outputs/figures", exist_ok=True)
    target_short = run_name.split('_')[-1]
    save_path = f"outputs/figures/3D_Relief_{target_short}_Quantum_{image_id[0].replace('.png', '')}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f">>> 3D Plot saved locally to {save_path}")

    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        api = HfApi(token=token)
        repo_id = "m-k-julkarnain/HQNN-Fetal-Head-Biometry"
        api.upload_file(path_or_fileobj=save_path, path_in_repo=f"visualizations/{os.path.basename(save_path)}", repo_id=repo_id)
        print(f">>> Uploaded 3D Relief to Hugging Face!")
    except Exception as e:
        print("HF Upload Skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True, choices=[1, 2, 3, 4], help="LODO Run Number (1-4)")
    args = parser.parse_args()
    generate_3d_surface(args.run)