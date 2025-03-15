import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class UCFCrimeDataset(Dataset):
    def __init__(self, abnormal_path, normal_path, resample=False):
        super().__init__()
        self.abnormal = get_all_videopaths(abnormal_path)
        self.normal = get_all_videopaths(normal_path)
        self.resample = resample
        self._update_data()

    def _update_data(self):
        if self.resample:
            # Lấy mẫu normal có số lượng = abnormal (Oversampling)
            sampled_normal = random.sample(self.normal, len(self.abnormal))
            self.data = np.concatenate((self.abnormal, sampled_normal), axis=0)
            self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(sampled_normal))), axis=0)
        else:
            # Giữ nguyên toàn bộ dữ liệu
            self.data = np.concatenate((self.abnormal, self.normal), axis=0)
            self.labels = np.concatenate((np.ones(len(self.abnormal)), np.zeros(len(self.normal))), axis=0)

        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def resample_data(self):
        """Gọi lại hàm này mỗi epoch để random lại tập normal khi train"""
        if self.resample:
            self._update_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_tensor = video_to_tensor(self.data[index])
        return video_tensor, self.labels[index]

def get_dataloader(abnormal_path, normal_path, batch_size, split_size=None, isTrain=False):
    num_workers = os.cpu_count() // 2

    if isTrain:
        # Tập train có resampling
        train_dataset = UCFCrimeDataset(abnormal_path, normal_path, resample=True)

        # Tập val giữ nguyên dữ liệu
        val_dataset = UCFCrimeDataset(abnormal_path, normal_path, resample=False)

        # Lấy chỉ số toàn bộ dữ liệu (val)
        indices = np.arange(len(val_dataset))
        labels = val_dataset.labels.numpy().flatten()

        # Chia train/val theo tỷ lệ giữ nguyên phân phối class
        train_idx, val_idx = train_test_split(indices, test_size=split_size, stratify=labels, random_state=42)

        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, train_dataset  # Trả về train_dataset để gọi resample mỗi epoch

    else:
        # Test lấy toàn bộ dữ liệu
        test_dataset = UCFCrimeDataset(abnormal_path, normal_path, resample=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader