import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_all_videopaths, video_to_tensor
import random

class UCFCrimeDataset(Dataset):
    def __init__(self, abnormal_path, normal_path):
        super().__init__()
        self.abnormal = get_all_videopaths(abnormal_path)
        self.normal = get_all_videopaths(normal_path)

        self.data = np.concatenate((self.abnormal, self.normal), axis=0)
        self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.normal))), axis=0)

        # self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def resample_data(self):
            normals = random.sample(self.normal, len(self.abnormal))
            self.data = np.concatenate((self.abnormal, normals), axis=0)
            self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(normals))), axis=0)
            self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
            
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        video_tensor = video_to_tensor(self.data[index])
        return video_tensor, self.labels[index]

def get_dataloader(abnormal_path, normal_path, batch_size, split_size=None):
    dataset = UCFCrimeDataset(abnormal_path, normal_path)
    num_workers = os.cpu_count() // 2  # Tận dụng đa luồng
    if split_size:
        # Lấy chỉ số của abnormal & normal
        indices = np.arange(len(dataset))
        labels = dataset.labels.numpy().flatten()

        # Chia tỉ lệ giữ nguyên phân phối class
        train_idx, val_idx = train_test_split(indices, test_size=split_size, stratify=labels, random_state=42) # 0.3 neu co validation
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)


        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        return train_loader, val_loader
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)