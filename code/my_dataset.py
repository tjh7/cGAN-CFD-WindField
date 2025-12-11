# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 16:42:06 2025

@author: Lenovo
"""



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class WindDataset(Dataset):
    def __init__(self, condition_csv, image_size=256):

        self.data = pd.read_csv(condition_csv, encoding='ansi')
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        building_img = Image.open(row['building']).convert('L')
        building_tensor = self.transform(building_img)
 
        windfield_img = Image.open(row['filename']).convert('RGB')
        windfield_tensor = self.transform(windfield_img)
        condition = torch.tensor([
            row['wind_speed_u(m/s)'],
            row['wind_speed_v(m/s)'],
            row['wind_speed_w(m/s)'],
            row['wind_dir']
        ], dtype=torch.float32)

        return {
            'building': building_tensor,      # [1, H, W]
            'windfield': windfield_tensor,    # [3, H, W]
            'condition': condition             # [4]
        }
