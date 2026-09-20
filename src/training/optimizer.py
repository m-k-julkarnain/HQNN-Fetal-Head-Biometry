""" Optimizer and Scheduler builder. """
import torch

def build_optimizer(model, learning_rate=1e-4, weight_decay=1e-2):
    q_parameters = []
    decay_parameters = []
    no_decay_parameters = []
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "pqc" in name.lower():
            q_parameters.append(parameter)
        elif parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
            
    optimizer = torch.optim.AdamW(
        [
            {"params": q_parameters, "lr": learning_rate * 10, "weight_decay": 0.0},
            {"params": decay_parameters, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "lr": learning_rate, "weight_decay": 0.0},
        ]
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