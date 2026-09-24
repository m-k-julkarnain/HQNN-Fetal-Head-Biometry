""" Fully Rewritten Generic Validation Script with Order-Aligned Test-Time Augmentation (TTA). """
import torch
import numpy as np
from tqdm.auto import tqdm
from configs.experiment import IMAGE_SIZE

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list, targets_list, image_ids_list = [], [], []
    
    with torch.no_grad():
        for images, coords, image_ids in tqdm(dataloader, desc="Validation", leave=False):
            images, coords = images.to(device), coords.to(device)
            
            # Pass 1: Normal Image
            outputs_1 = model(images)
            preds_1 = outputs_1[0] if isinstance(outputs_1, tuple) else outputs_1
            
            # Pass 2: TTA Horizontal Flip
            images_hf = torch.flip(images, dims=[3])
            outputs_2 = model(images_hf)
            preds_2 = outputs_2[0] if isinstance(outputs_2, tuple) else outputs_2
            
            # Mathematically unflip coordinates (Invert X in [-1, 1] space)
            preds_2_unflipped = preds_2.clone()
            preds_2_unflipped[:, 0::2] = -preds_2[:, 0::2]
            
            # TTA Order Alignment for independent biometry pairs (Diameters 1 and 2)
            # Diameter 1 (Points 0 & 1 -> indices 0:4)
            p1_d1, p2_d1 = preds_1[:, :4], preds_2_unflipped[:, :4]
            dist_straight = torch.norm(p1_d1 - p2_d1, dim=1)
            p2_d1_swapped = torch.cat([p2_d1[:, 2:4], p2_d1[:, 0:2]], dim=1)
            dist_cross = torch.norm(p1_d1 - p2_d1_swapped, dim=1)
            p2_d1_aligned = torch.where((dist_cross < dist_straight).unsqueeze(1), p2_d1_swapped, p2_d1)
            
            # Diameter 2 (Points 2 & 3 -> indices 4:8)
            p1_d2, p2_d2 = preds_1[:, 4:], preds_2_unflipped[:, 4:]
            dist_straight_2 = torch.norm(p1_d2 - p2_d2, dim=1)
            p2_d2_swapped = torch.cat([p2_d2[:, 2:4], p2_d2[:, 0:2]], dim=1)
            dist_cross_2 = torch.norm(p1_d2 - p2_d2_swapped, dim=1)
            p2_d2_aligned = torch.where((dist_cross_2 < dist_straight_2).unsqueeze(1), p2_d2_swapped, p2_d2)
            
            # Average the Order-Aligned TTA predictions
            preds_2_aligned = torch.cat([p2_d1_aligned, p2_d2_aligned], dim=1)
            preds = (preds_1 + preds_2_aligned) / 2.0
            
            # Compute Loss
            loss_val = criterion((preds, outputs_1[1]), coords) if isinstance(outputs_1, tuple) else criterion(preds, coords)
            running_loss += loss_val.item()
            
            preds_list.extend(preds.cpu().numpy().tolist())
            targets_list.extend(coords.cpu().numpy().tolist())
            image_ids_list.extend(image_ids)
            
    # Denormalize from [-1, 1] to pixel space [0, IMAGE_SIZE]
    preds_arr = ((np.array(preds_list) + 1.0) / 2.0) * IMAGE_SIZE
    targets_arr = ((np.array(targets_list) + 1.0) / 2.0) * IMAGE_SIZE
    
    # Diameter 1 Error (e.g. TAD / BPD)
    mae_d1_straight = np.mean(np.abs(preds_arr[:, :4] - targets_arr[:, :4]), axis=1)
    targets_d1_flipped = np.concatenate([targets_arr[:, 2:4], targets_arr[:, 0:2]], axis=1)
    mae_d1_flipped = np.mean(np.abs(preds_arr[:, :4] - targets_d1_flipped), axis=1)
    mae_d1 = np.minimum(mae_d1_straight, mae_d1_flipped)
    
    # Diameter 2 Error (e.g. APAD / OFD)
    mae_d2_straight = np.mean(np.abs(preds_arr[:, 4:] - targets_arr[:, 4:]), axis=1)
    targets_d2_flipped = np.concatenate([targets_arr[:, 6:8], targets_arr[:, 4:6]], axis=1)
    mae_d2_flipped = np.mean(np.abs(preds_arr[:, 4:] - targets_d2_flipped), axis=1)
    mae_d2 = np.minimum(mae_d2_straight, mae_d2_flipped)
    
    true_mae_per_image = (mae_d1 + mae_d2) / 2.0
    mean_mae = np.mean(true_mae_per_image)
    
    detailed_records = []
    for i in range(len(image_ids_list)):
        detailed_records.append({
            "Patient_ID": image_ids_list[i],
            "Diameter1_Error_px": round(float(mae_d1[i]), 4),
            "Diameter2_Error_px": round(float(mae_d2[i]), 4),
            "Total_MAE_px": round(float(true_mae_per_image[i]), 4)
        })
        
    return running_loss / len(dataloader), detailed_records, {"val_loss": running_loss / len(dataloader), "mean_mae": float(mean_mae)}