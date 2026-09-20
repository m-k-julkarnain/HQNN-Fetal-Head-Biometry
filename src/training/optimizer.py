""" Optimizer and Scheduler builder. """
import torch

def build_optimizer(model, learning_rate=1e-4, weight_decay=1e-2):
    decay_parameters = []
    no_decay_parameters = []
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
            
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    return optimizer

def build_scheduler(optimizer, epochs, steps_per_epoch, eta_min=1e-6):
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=eta_min
    )
    return scheduler