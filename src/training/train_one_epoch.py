import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler=None, scaler=None, device=None):
    model.train()
    running_loss = 0.0
    domain_criterion = nn.CrossEntropyLoss()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for i, (images, coords, _) in enumerate(progress_bar):
        images, coords = images.to(device, non_blocking=True), coords.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        device_type = device.type if hasattr(device, 'type') else 'cuda'
        
        # Dynamic GRL Alpha scaling
        p = float(i) / len(dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # Generate dummy domain labels (Half batch Domain 0, Half batch Domain 1)
        # This forces the discriminator to actively balance the feature space
        batch_size = images.size(0)
        domain_labels = torch.cat([torch.zeros(batch_size // 2, dtype=torch.long), 
                                   torch.ones(batch_size - batch_size // 2, dtype=torch.long)]).to(device)
        
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            outputs = model(images, alpha)
            if len(outputs) == 3: # HQ-DANN (Quantum)
                coords_pred, mask_logits, domain_logits = outputs
                biometry_loss = criterion((coords_pred, mask_logits), coords)
                domain_loss = domain_criterion(domain_logits, domain_labels)
                loss = biometry_loss + (0.1 * domain_loss)
            else: # Classical Baseline
                loss = criterion(outputs, coords)
            
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not (scaler.get_scale() < scale_before) and scheduler is not None: scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
            
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(dataloader)