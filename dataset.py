import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import get_all_videopaths, video_to_tensor

class UCFCrimeDataset(Dataset):
    def __init__(self, abnormal_path, normal_path):
        super().__init__()
        abnormal = get_all_videopaths(abnormal_path)
        normal = get_all_videopaths(normal_path)

        self.data = np.concatenate((abnormal, normal), axis=0)
        self.labels = np.concatenate((np.ones(len(abnormal)), np.zeros(len(normal))), axis=0)

        # self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_tensor = video_to_tensor(self.data[index])
        return video_tensor, self.labels[index]
    
def pad_video(video_tensor, target_frames=32):
    """Đệm thêm frame nếu video chưa đủ target_frames."""
    C, T, H, W = video_tensor.shape  # Lấy số frame T hiện tại

    if T >= target_frames:
        return video_tensor[:, :target_frames, :, :]  # Cắt bớt nếu quá dài

    # Tạo tensor đệm (sao chép frame cuối cùng)
    pad_size = target_frames - T
    pad_tensor = video_tensor[:, -1:, :, :].repeat(1, pad_size, 1, 1)  # (C, pad_size, H, W)

    return torch.cat([video_tensor, pad_tensor], dim=1)  # Ghép vào cuối theo trục T

def custom_collate_fn(batch):
    videos, labels = zip(*batch)  # Tách dữ liệu

    # Đệm tất cả video
    padded_videos = [pad_video(v) for v in videos]

    return torch.stack(padded_videos), torch.stack(labels)

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
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
            num_workers=num_workers, pin_memory=True
        )

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        return train_loader, val_loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn,
                      num_workers=num_workers, pin_memory=True)