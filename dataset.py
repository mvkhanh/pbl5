import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os
import torch
from torch.utils.data import Dataset
from utils.utils import get_all_videopaths, video_to_tensor
import random

class UCFCrimeDataset(Dataset):
    def __init__(self, path, isTrain):
        super().__init__()
        abnormal_path = os.path.join('Abnormal')
        normal_path = os.path.join('NormalVideos')
        self.abnormal = get_all_videopaths(abnormal_path)
        self.normal = get_all_videopaths(normal_path)
        self.isTrain = isTrain
        
        self.data = np.concatenate((self.abnormal, self.normal), axis=0)
        self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.normal))), axis=0)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

        # Dữ liệu được resample (dùng cho training)
        if isTrain:
            self.resample_data()

    def resample_data(self):
        # Lấy ngẫu nhiên 1 phần normal bằng với abnormal
        self.sampled_normals = random.sample(self.normal, len(self.abnormal))
        self.data = np.concatenate((self.abnormal, self.sampled_normals), axis=0)
        self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.sampled_normals))), axis=0)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_path = self.data[index]
        label = self.labels[index]
        
        video_tensor = video_to_tensor(video_path)
        return video_tensor, label

def get_dataloader(path, batch_size, isTrain=False):
    dataset = UCFCrimeDataset(path, isTrain)
    num_workers = os.cpu_count() // 2  # Tận dụng đa luồng    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=isTrain,
                            num_workers=num_workers, pin_memory=True)
       
    print(f"Dataset size: {len(dataset)}")
    return data_loader