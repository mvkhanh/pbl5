import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os
import torch
from torch.utils.data import Dataset
from utils.utils import get_all_videopaths, video_to_tensor
import random

class UCFCrimeDataset(Dataset):
    def __init__(self, abnormal_path, normal_path, resample=False):
        super().__init__()
        self.abnormal = get_all_videopaths(abnormal_path)
        self.normal = get_all_videopaths(normal_path)
        self.resample = resample

        # Dữ liệu gốc (dùng cho validation)
        self.full_data = np.concatenate((self.abnormal, self.normal), axis=0)
        self.full_labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.normal))), axis=0)

        # Dữ liệu được resample (dùng cho training)
        self.resample_data()

    def resample_data(self):
        # Lấy ngẫu nhiên 1 phần normal bằng với abnormal
        self.sampled_normals = random.sample(self.normal, len(self.abnormal))
        self.data = np.concatenate((self.abnormal, self.sampled_normals), axis=0)
        self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.sampled_normals))), axis=0)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data) if self.resample else len(self.full_data)

    def __getitem__(self, index):
        if self.resample:
            video_path = self.data[index]
            label = self.labels[index]
        else:
            video_path = self.full_data[index]
            label = self.full_labels[index]
        
        video_tensor = video_to_tensor(video_path)
        return video_tensor, label

def get_dataloader(abnormal_path, normal_path, batch_size, split_size=None):
    dataset = UCFCrimeDataset(abnormal_path, normal_path)
    num_workers = os.cpu_count() // 2  # Tận dụng đa luồng
    if split_size:
         # Dataset train có resample
        train_dataset = UCFCrimeDataset(abnormal_path, normal_path, resample=True)
    
        # Dataset validation không resample
        val_dataset = UCFCrimeDataset(abnormal_path, normal_path, resample=False)

        if split_size:
            indices = np.arange(len(val_dataset))
            labels = val_dataset.full_labels

            # Chia tỉ lệ giữ nguyên phân phối class
            train_idx, val_idx = train_test_split(indices, test_size=split_size, stratify=labels, random_state=42)
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        return train_loader, val_loader
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)