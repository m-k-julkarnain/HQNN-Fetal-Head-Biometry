import random
import math
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class FetalHeadDataset(Dataset):
    def __init__(self, csv_file, image_dir, image_size=224):
        self.data = pd.read_csv(Path(csv_file))
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.is_train = 'train' in str(csv_file).lower()
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        id_col = next((c for c in ['image_name', 'image_id', 'id_code', 'filename', 'ID'] if c in self.data.columns), None)
        image_id = str(row[id_col]) if id_col else str(row.iloc[0])
        
        img_path = self.image_dir / (image_id if image_id.endswith(('.png','.jpg','.jpeg')) else f"{image_id}.jpg")
        if not img_path.exists():
            img_path = self.image_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = self.image_dir / f"{image_id}.jpeg"
            
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        c_map = {str(c).lower(): c for c in self.data.columns}
        
        bpd_keys = ['bpd_1_x', 'bpd_1_y', 'bpd_2_x', 'bpd_2_y']
        bpd_raw = [float(row[c_map.get(k, c_map.get(k.upper()))]) for k in bpd_keys if k in c_map or k.upper() in c_map]
        
        ofd_keys = ['ofd_1_x', 'ofd_1_y', 'ofd_2_x', 'ofd_2_y']
        if all(k in c_map or k.upper() in c_map for k in ofd_keys):
            ofd_raw = [float(row[c_map.get(k, c_map.get(k.upper()))]) for k in ofd_keys]
        else:
            ax1, ay1, ax2, ay2 = bpd_raw
            cx, cy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
            dx, dy = ax2 - ax1, ay2 - ay1
            length = math.sqrt(dx**2 + dy**2) + 1e-6
            nx, ny = -dy / length, dx / length
            ofd_raw = [cx - nx * (length * 0.6), cy - ny * (length * 0.6), cx + nx * (length * 0.6), cy + ny * (length * 0.6)]

        # Map to standard square resolution before rotation to preserve aspect ratio
        pts = [
            (bpd_raw[0] / w) * self.image_size, (bpd_raw[1] / h) * self.image_size,
            (bpd_raw[2] / w) * self.image_size, (bpd_raw[3] / h) * self.image_size,
            (ofd_raw[0] / w) * self.image_size, (ofd_raw[1] / h) * self.image_size,
            (ofd_raw[2] / w) * self.image_size, (ofd_raw[3] / h) * self.image_size
        ]
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Synchronized rotation in pixel space around exact center
        if self.is_train and random.random() > 0.4:
            angle_deg = random.uniform(-20, 20)
            image = TF.rotate(image, angle_deg)
            
            # Counter-clockwise rotation matching TF.rotate in image coordinates
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            center = self.image_size / 2.0
            
            def rotate_px(px, py):
                x = px - center
                y = py - center
                # Standard screen rotation matrix for TF.rotate
                rx = x * cos_a - y * sin_a + center
                ry = x * sin_a + y * cos_a + center
                return max(0.0, min(float(self.image_size), rx)), max(0.0, min(float(self.image_size), ry))
                
            pts[0], pts[1] = rotate_px(pts[0], pts[1])
            pts[2], pts[3] = rotate_px(pts[2], pts[3])
            pts[4], pts[5] = rotate_px(pts[4], pts[5])
            pts[6], pts[7] = rotate_px(pts[6], pts[7])

        # Normalize coordinates strictly to [-1, 1]
        norm_coords = [(p / self.image_size) * 2.0 - 1.0 for p in pts]
        coords = torch.tensor(norm_coords, dtype=torch.float32)

        # Apply photometrics and normalization
        tensor_img = TF.to_tensor(self.color_jitter(image))
        tensor_img = self.normalize(tensor_img)

        return tensor_img, coords, image_id