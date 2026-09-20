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
        self.is_train = 'train' in str(csv_file).lower()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

        x1_b = (bpd_raw[0] / w) * 2.0 - 1.0
        y1_b = (bpd_raw[1] / h) * 2.0 - 1.0
        x2_b = (bpd_raw[2] / w) * 2.0 - 1.0
        y2_b = (bpd_raw[3] / h) * 2.0 - 1.0

        x1_o = (ofd_raw[0] / w) * 2.0 - 1.0
        y1_o = (ofd_raw[1] / h) * 2.0 - 1.0
        x2_o = (ofd_raw[2] / w) * 2.0 - 1.0
        y2_o = (ofd_raw[3] / h) * 2.0 - 1.0

        # Heavy Spatial Augmentation: Scale, Translation, and Rotation
        if self.is_train and random.random() > 0.3:
            angle_deg = random.uniform(-25, 25)
            scale = random.uniform(0.85, 1.15)
            tx_px = random.uniform(-0.1, 0.1) * w
            ty_px = random.uniform(-0.1, 0.1) * h
            
            image = TF.affine(image, angle=angle_deg, translate=(int(tx_px), int(ty_px)), scale=scale, shear=0)
            
            a_rad = math.radians(angle_deg)
            ca, sa = math.cos(a_rad), math.sin(a_rad)
            
            def affine_pt(x_norm, y_norm):
                nx = (x_norm * ca + y_norm * sa) * scale
                ny = (-x_norm * sa + y_norm * ca) * scale
                nx += (tx_px / w) * 2.0
                ny += (ty_px / h) * 2.0
                return max(-1.0, min(1.0, nx)), max(-1.0, min(1.0, ny))
                
            x1_b, y1_b = affine_pt(x1_b, y1_b)
            x2_b, y2_b = affine_pt(x2_b, y2_b)
            x1_o, y1_o = affine_pt(x1_o, y1_o)
            x2_o, y2_o = affine_pt(x2_o, y2_o)

        coords = torch.tensor([x1_b, y1_b, x2_b, y2_b, x1_o, y1_o, x2_o, y2_o], dtype=torch.float32)

        return self.transform(image), coords, image_id