""" Validation script for Fetal Head Biometry HQNN (Order-Invariant & CDF Ready). """
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
            outputs = model(images)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs
            
            if preds.ndim > 2:
                preds = preds.view(preds.size(0), -1)
                
            loss_val = criterion(outputs, coords).item() if isinstance(outputs, tuple) else criterion(preds, coords).item()
            running_loss += loss_val
            
            preds_list.extend(preds.cpu().numpy().tolist())
            targets_list.extend(coords.cpu().numpy().tolist())
            image_ids_list.extend(image_ids)
            
    preds_arr = ((np.array(preds_list) + 1.0) / 2.0) * IMAGE_SIZE
    targets_arr = ((np.array(targets_list) + 1.0) / 2.0) * IMAGE_SIZE
    
    # Order-Invariant BPD Error
    mae_bpd_straight = np.mean(np.abs(preds_arr[:, :4] - targets_arr[:, :4]), axis=1)
    targets_bpd_flipped = np.concatenate([targets_arr[:, 2:4], targets_arr[:, 0:2]], axis=1)
    mae_bpd_flipped = np.mean(np.abs(preds_arr[:, :4] - targets_bpd_flipped), axis=1)
    mae_bpd = np.minimum(mae_bpd_straight, mae_bpd_flipped)
    
    # Order-Invariant OFD Error
    mae_ofd_straight = np.mean(np.abs(preds_arr[:, 4:] - targets_arr[:, 4:]), axis=1)
    targets_ofd_flipped = np.concatenate([targets_arr[:, 6:8], targets_arr[:, 4:6]], axis=1)
    mae_ofd_flipped = np.mean(np.abs(preds_arr[:, 4:] - targets_ofd_flipped), axis=1)
    mae_ofd = np.minimum(mae_ofd_straight, mae_ofd_flipped)
    
    true_mae_per_image = (mae_bpd + mae_ofd) / 2.0
    mean_mae = np.mean(true_mae_per_image)
    
    # Rich scan-by-scan record for CDF and 3D plotting
    detailed_records = []
    for i in range(len(image_ids_list)):
        detailed_records.append({
            "Patient_ID": image_ids_list[i],
            "BPD_Error_px": round(float(mae_bpd[i]), 4),
            "OFD_Error_px": round(float(mae_ofd[i]), 4),
            "Total_MAE_px": round(float(true_mae_per_image[i]), 4)
        })
        
    return running_loss / len(dataloader), detailed_records, {"val_loss": running_loss / len(dataloader), "mean_mae": float(mean_mae)}