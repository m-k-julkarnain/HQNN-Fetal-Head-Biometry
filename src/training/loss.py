""" Loss functions for Fetal Head Biometry HQNN. """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.experiment import IMAGE_SIZE

class WingLoss(nn.Module):
    def __init__(self, w=0.05, epsilon=0.01):
        super().__init__()
        self.w, self.epsilon = w, epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, preds, targets):
        x = torch.abs(preds - targets)
        loss = torch.where(x < self.w, self.w * torch.log(1.0 + x / self.epsilon), x - self.c)
        return loss.mean(dim=1)

class StructureAwareBiometryLoss(nn.Module):
    def __init__(self, w=0.05, epsilon=0.01, mask_weight=0.4, boundary_weight=0.2, distal_weight=0.15):
        super().__init__()
        self.wing = WingLoss(w, epsilon)
        self.alpha = 0.3
        self.gamma = 0.15
        self.mask_weight = mask_weight
        self.boundary_weight = boundary_weight
        self.distal_weight = distal_weight
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

    def compute_single_diameter_loss(self, p, t):
        ep_loss = self.wing(p, t)
        p_y1, p_y2 = p[:, 1], p[:, 3]
        t_y1, t_y2 = t[:, 1], t[:, 3]
        is_p2_bottom = (t_y2 > t_y1).float()
        
        err_y1 = self.wing(p_y1.unsqueeze(1), t_y1.unsqueeze(1))
        err_y2 = self.wing(p_y2.unsqueeze(1), t_y2.unsqueeze(1))
        distal_loss = (1.0 - is_p2_bottom) * err_y1 + is_p2_bottom * err_y2
        
        p_mid = torch.stack([(p[:, 0] + p[:, 2]) / 2.0, (p[:, 1] + p[:, 3]) / 2.0], dim=1)
        t_mid = torch.stack([(t[:, 0] + t[:, 2]) / 2.0, (t[:, 1] + t[:, 3]) / 2.0], dim=1)
        
        p_vec = p[:, 2:4] - p[:, 0:2]
        t_vec = t[:, 2:4] - t[:, 0:2]
        angle_loss = 1.0 - F.cosine_similarity(p_vec, t_vec, dim=1)
        
        return ep_loss + (self.alpha * self.wing(p_mid, t_mid)) + (self.gamma * angle_loss) + (self.distal_weight * distal_loss)

    def forward(self, model_outputs, targets):
        if isinstance(model_outputs, tuple):
            preds, pred_mask = model_outputs
        else:
            preds = model_outputs
            pred_mask = None

        # Split 8 coordinates into BPD (first 4) and OFD (last 4)
        preds_bpd = preds[:, :4]
        preds_ofd = preds[:, 4:]
        targets_bpd = targets[:, :4]
        targets_ofd = targets[:, 4:]

        loss_bpd = self.compute_single_diameter_loss(preds_bpd, targets_bpd).mean()
        loss_ofd = self.compute_single_diameter_loss(preds_ofd, targets_ofd).mean()
        biometry_loss = 0.5 * loss_bpd + 0.5 * loss_ofd

        # Fetal head anatomical circumference constraint
        if pred_mask is not None:
            batch_size = targets.shape[0]
            device = targets.device
            target_mask = torch.zeros((batch_size, 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
            
            coords_px = ((targets_bpd + 1.0) / 2.0) * float(IMAGE_SIZE)
            y_coords = torch.linspace(0, IMAGE_SIZE - 1, IMAGE_SIZE, device=device).view(1, 1, IMAGE_SIZE, 1)
            x_coords = torch.linspace(0, IMAGE_SIZE - 1, IMAGE_SIZE, device=device).view(1, 1, 1, IMAGE_SIZE)
            
            for b in range(batch_size):
                x1, y1, x2, y2 = coords_px[b]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                radius = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2.0
                radius = torch.clamp(radius, min=10.0, max=100.0)
                
                dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
                target_mask[b, 0] = (dist_sq <= radius**2).float()
            
            mask_loss = self.bce_logits_loss(pred_mask, target_mask)
            return biometry_loss + (self.mask_weight * mask_loss)
            
        return biometry_loss

def build_loss():
    return StructureAwareBiometryLoss()