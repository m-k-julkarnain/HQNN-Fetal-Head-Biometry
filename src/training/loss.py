# """ Order-Invariant Pixel-Space Loss with Gaussian Soft-Masking. """
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from configs.experiment import IMAGE_SIZE

# class WingLoss(nn.Module):
#     def __init__(self, w=10.0, epsilon=2.0):
#         super().__init__()
#         self.w, self.epsilon = w, epsilon
#         self.c = w - w * math.log(1.0 + w / epsilon)

#     def forward(self, preds, targets):
#         x = torch.abs(preds - targets)
#         return torch.where(x < self.w, self.w * torch.log(1.0 + x / self.epsilon), x - self.c)

# class StructureAwareBiometryLoss(nn.Module):
#     def __init__(self, w=10.0, epsilon=2.0, mask_weight=0.5, boundary_weight=0.2, distal_weight=0.15):
#         super().__init__()
#         self.wing = WingLoss(w, epsilon)
#         self.alpha = 0.5
#         self.gamma = 0.2
#         self.mask_weight = mask_weight
#         self.bce_logits_loss = nn.BCEWithLogitsLoss()

#     def compute_order_invariant_loss(self, p, t):
#         p_px = ((p + 1.0) / 2.0) * float(IMAGE_SIZE)
#         t_px = ((t + 1.0) / 2.0) * float(IMAGE_SIZE)
        
#         loss_straight = self.wing(p_px, t_px).mean(dim=1)
#         t_flipped_px = torch.cat([t_px[:, 2:4], t_px[:, 0:2]], dim=1)
#         loss_flipped = self.wing(p_px, t_flipped_px).mean(dim=1)
#         ep_loss = torch.minimum(loss_straight, loss_flipped)
        
#         p_mid = torch.stack([(p_px[:, 0] + p_px[:, 2]) / 2.0, (p_px[:, 1] + p_px[:, 3]) / 2.0], dim=1)
#         t_mid = torch.stack([(t_px[:, 0] + t_px[:, 2]) / 2.0, (t_px[:, 1] + t_px[:, 3]) / 2.0], dim=1)
#         mid_loss = self.wing(p_mid, t_mid).mean(dim=1)
        
#         p_vec = p[:, 2:4] - p[:, 0:2]
#         t_vec = t[:, 2:4] - t[:, 0:2]
#         angle_loss = 1.0 - torch.abs(F.cosine_similarity(p_vec, t_vec, dim=1))
        
#         return ep_loss + (self.alpha * mid_loss) + (self.gamma * angle_loss * 50.0)

#     def forward(self, model_outputs, targets):
#         if isinstance(model_outputs, tuple):
#             preds, pred_mask = model_outputs
#             if len(model_outputs) == 3:
#                 preds, pred_mask, _ = model_outputs # Handle DANN logits if present
#         else:
#             preds = model_outputs
#             pred_mask = None

#         preds_bpd, preds_ofd = preds[:, :4], preds[:, 4:]
#         targets_bpd, targets_ofd = targets[:, :4], targets[:, 4:]

#         loss_bpd = self.compute_order_invariant_loss(preds_bpd, targets_bpd).mean()
#         loss_ofd = self.compute_order_invariant_loss(preds_ofd, targets_ofd).mean()
#         biometry_loss = 0.5 * loss_bpd + 0.5 * loss_ofd

#         if pred_mask is not None:
#             batch_size = targets.shape[0]
#             device = targets.device
#             target_mask = torch.zeros((batch_size, 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
            
#             coords_px = ((targets_bpd + 1.0) / 2.0) * float(IMAGE_SIZE)
#             y_coords = torch.linspace(0, IMAGE_SIZE - 1, IMAGE_SIZE, device=device).view(1, 1, IMAGE_SIZE, 1)
#             x_coords = torch.linspace(0, IMAGE_SIZE - 1, IMAGE_SIZE, device=device).view(1, 1, 1, IMAGE_SIZE)
            
#             for b in range(batch_size):
#                 x1, y1, x2, y2 = coords_px[b]
#                 cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
#                 # Bivariate Gaussian Soft-Mask for Uncertainty Calibration
#                 sigma_x = torch.clamp(torch.abs(x2 - x1) / 4.0, min=15.0)
#                 sigma_y = torch.clamp(torch.abs(y2 - y1) / 4.0, min=15.0)
                
#                 dist_sq = ((x_coords - cx)**2) / (2 * sigma_x**2) + ((y_coords - cy)**2) / (2 * sigma_y**2)
#                 target_mask[b, 0] = torch.exp(-dist_sq) # Smooth probability dome [0, 1]
            
#             mask_loss = self.bce_logits_loss(pred_mask, target_mask)
#             # Reduced multiplier to prevent gradient explosion on the soft mask
#             return biometry_loss + (self.mask_weight * mask_loss * 15.0)
            
#         return biometry_loss

# def build_loss():
#     return StructureAwareBiometryLoss()


import torch
import torch.nn as nn
import torch.nn.functional as F

class FetalBiometryLoss(nn.Module):
    def __init__(self, smooth_l1_weight=1.0, distance_weight=1.0):
        super(FetalBiometryLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        self.l1_weight = smooth_l1_weight
        self.dist_weight = distance_weight

    def forward(self, pred_coords, true_coords):
        """
        Handles both tensor and tuple model outputs.
        pred_coords: Tensor or Tuple containing coordinate predictions.
        true_coords: Tensor of shape [Batch, 8]
        """
        if isinstance(pred_coords, tuple):
            pred_coords = pred_coords[0]
            
        # 1. Base regression loss
        reg_loss = self.smooth_l1(pred_coords, true_coords)
        
        # 2. Geometric Euclidean distance penalty across individual landmark pairs
        pred_pts = pred_coords.view(-1, 4, 2)
        true_pts = true_coords.view(-1, 4, 2)
        
        euclidean_dists = torch.sqrt(torch.sum((pred_pts - true_pts) ** 2, dim=-1) + 1e-6)
        dist_loss = torch.mean(euclidean_dists)
        
        # 3. Diameter structural length consistency penalty (TAD and APAD lengths)
        pred_tad_len = torch.sqrt(torch.sum((pred_pts[:, 0, :] - pred_pts[:, 1, :]) ** 2, dim=-1) + 1e-6)
        true_tad_len = torch.sqrt(torch.sum((true_pts[:, 0, :] - true_pts[:, 1, :]) ** 2, dim=-1) + 1e-6)
        tad_len_loss = torch.mean(torch.abs(pred_tad_len - true_tad_len))
        
        pred_apad_len = torch.sqrt(torch.sum((pred_pts[:, 2, :] - pred_pts[:, 3, :]) ** 2, dim=-1) + 1e-6)
        true_apad_len = torch.sqrt(torch.sum((true_pts[:, 2, :] - true_pts[:, 3, :]) ** 2, dim=-1) + 1e-6)
        apad_len_loss = torch.mean(torch.abs(pred_apad_len - true_apad_len))
        
        total_loss = (self.l1_weight * reg_loss) + \
                     (self.dist_weight * dist_loss) + \
                     (0.5 * (tad_len_loss + apad_len_loss))
                     
        return total_loss

def build_loss():
    return FetalBiometryLoss()