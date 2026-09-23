import random
import math
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class FetalAbdomenDataset(Dataset):
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
        
        # Abdomen Keys: TAD and APAD
        tad_keys = ['tad_1_x', 'tad_1_y', 'tad_2_x', 'tad_2_y']
        apad_keys = ['apad_1_x', 'apad_1_y', 'apad_2_x', 'apad_2_y']
        
        tad_raw = [float(row[c_map.get(k, c_map.get(k.upper()))]) for k in tad_keys if k in c_map or k.upper() in c_map]
        apad_raw = [float(row[c_map.get(k, c_map.get(k.upper()))]) for k in apad_keys if k in c_map or k.upper() in c_map]
        
        # Map to standard square resolution before rotation to preserve aspect ratio
        pts = [
            (tad_raw[0] / w) * self.image_size, (tad_raw[1] / h) * self.image_size,
            (tad_raw[2] / w) * self.image_size, (tad_raw[3] / h) * self.image_size,
            (apad_raw[0] / w) * self.image_size, (apad_raw[1] / h) * self.image_size,
            (apad_raw[2] / w) * self.image_size, (apad_raw[3] / h) * self.image_size
        ]
        
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Synchronized rotation in pixel space around exact center
        if self.is_train and random.random() > 0.4:
            angle_deg = random.uniform(-20, 20)
            image = TF.rotate(image, angle_deg)
            
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            center = self.image_size / 2.0
            
            def rotate_px(px, py):
                x = px - center
                y = py - center
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
        
        tensor_img = TF.to_tensor(self.color_jitter(image))
        tensor_img = self.normalize(tensor_img)
        
        return tensor_img, coords, image_id