import torch
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler=None, scaler=None, device=None):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, coords, _ in progress_bar:
        images, coords = images.to(device, non_blocking=True), coords.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        device_type = device.type if hasattr(device, 'type') else 'cuda'
        
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            loss = criterion(model(images), coords)
            
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