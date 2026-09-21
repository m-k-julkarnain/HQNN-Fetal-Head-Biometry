""" Validation script with Order-Aligned Test-Time Augmentation (TTA). """
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
            
            # Mathematically unflip coordinates (Invert X)
            preds_2_unflipped = preds_2.clone()
            preds_2_unflipped[:, 0::2] = -preds_2[:, 0::2]
            
            # TTA Order Alignment: Because our loss is order-invariant, the model might swap endpoints on flipped images!
            # We measure distance to ensure we average Left with Left, not Left with Right.
            
            # BPD Alignment
            p1_bpd, p2_bpd = preds_1[:, :4], preds_2_unflipped[:, :4]
            dist_straight = torch.norm(p1_bpd - p2_bpd, dim=1)
            p2_bpd_swapped = torch.cat([p2_bpd[:, 2:4], p2_bpd[:, 0:2]], dim=1)
            dist_cross = torch.norm(p1_bpd - p2_bpd_swapped, dim=1)
            p2_bpd_aligned = torch.where((dist_cross < dist_straight).unsqueeze(1), p2_bpd_swapped, p2_bpd)
            
            # OFD Alignment
            p1_ofd, p2_ofd = preds_1[:, 4:], preds_2_unflipped[:, 4:]
            dist_straight_o = torch.norm(p1_ofd - p2_ofd, dim=1)
            p2_ofd_swapped = torch.cat([p2_ofd[:, 2:4], p2_ofd[:, 0:2]], dim=1)
            dist_cross_o = torch.norm(p1_ofd - p2_ofd_swapped, dim=1)
            p2_ofd_aligned = torch.where((dist_cross_o < dist_straight_o).unsqueeze(1), p2_ofd_swapped, p2_ofd)
            
            # Average the Order-Aligned TTA predictions
            preds_2_aligned = torch.cat([p2_bpd_aligned, p2_ofd_aligned], dim=1)
            preds = (preds_1 + preds_2_aligned) / 2.0
            
            # Compute Loss (pass only tuple format to criterion)
            loss_val = criterion((preds, outputs_1[1]), coords) if isinstance(outputs_1, tuple) else criterion(preds, coords)
            running_loss += loss_val.item()
            
            preds_list.extend(preds.cpu().numpy().tolist())
            targets_list.extend(coords.cpu().numpy().tolist())
            image_ids_list.extend(image_ids)
            
    preds_arr = ((np.array(preds_list) + 1.0) / 2.0) * IMAGE_SIZE
    targets_arr = ((np.array(targets_list) + 1.0) / 2.0) * IMAGE_SIZE
    
    mae_bpd_straight = np.mean(np.abs(preds_arr[:, :4] - targets_arr[:, :4]), axis=1)
    targets_bpd_flipped = np.concatenate([targets_arr[:, 2:4], targets_arr[:, 0:2]], axis=1)
    mae_bpd_flipped = np.mean(np.abs(preds_arr[:, :4] - targets_bpd_flipped), axis=1)
    mae_bpd = np.minimum(mae_bpd_straight, mae_bpd_flipped)
    
    mae_ofd_straight = np.mean(np.abs(preds_arr[:, 4:] - targets_arr[:, 4:]), axis=1)
    targets_ofd_flipped = np.concatenate([targets_arr[:, 6:8], targets_arr[:, 4:6]], axis=1)
    mae_ofd_flipped = np.mean(np.abs(preds_arr[:, 4:] - targets_ofd_flipped), axis=1)
    mae_ofd = np.minimum(mae_ofd_straight, mae_ofd_flipped)
    
    true_mae_per_image = (mae_bpd + mae_ofd) / 2.0
    mean_mae = np.mean(true_mae_per_image)
    
    detailed_records = []
    for i in range(len(image_ids_list)):
        detailed_records.append({
            "Patient_ID": image_ids_list[i],
            "BPD_Error_px": round(float(mae_bpd[i]), 4),
            "OFD_Error_px": round(float(mae_ofd[i]), 4),
            "Total_MAE_px": round(float(true_mae_per_image[i]), 4)
        })
        
    return running_loss / len(dataloader), detailed_records, {"val_loss": running_loss / len(dataloader), "mean_mae": float(mean_mae)}