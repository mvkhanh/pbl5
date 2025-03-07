import os
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
from uniformer import uniformer_base
from huggingface_hub import hf_hub_download
from transforms import (
    GroupCenterCrop,
    ToTorchFormatTensor,
    GroupNormalize,
    Stack,
    GroupScale
)

# Cấu hình
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Số segment xử lý cùng lúc
SEGMENT_DIR = "5fps_Segments/Train"  # Thư mục chứa các segment video
OUTPUT_DIR = "Segment_Features_tmp"  # Nơi lưu đặc trưng

# Load model Uniformer
model = uniformer_base()
model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device).eval()

# Transform
crop_size = 224
scale_size = 256
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

transform = T.Compose([
    GroupScale(scale_size),
    GroupCenterCrop(crop_size),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std) 
])

def extract_features(model, batch_segments):
    """ Trích xuất đặc trưng từ batch segment """
    with torch.no_grad():
        batch_segments = batch_segments.to(device)  # Chuyển sang GPU
        features = model.forward_features(batch_segments)
        gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        features = gap(features).view(features.shape[0], -1)  # [B, C, 1, 1, 1] -> [B, C]
    return features.cpu().numpy()

def video_to_tensor(video_path):
    """ Đọc video và chuyển thành tensor có shape [3, T, H, W] """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    cap.release()
    
    if len(frames) != 32:
        print(f"⚠️ {video_path}: không đủ 32 frames ({len(frames)})")
        return None
    
    # Áp dụng transform
    frames = transform(frames)  # [3, 32, 224, 224]
    return frames

def process_segments():
    """ Xử lý từng batch ngay khi đủ số lượng """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    batch_segments = []

    for root, _, files in os.walk(SEGMENT_DIR):
        for file in sorted(files):
            if not file.endswith(".mp4"):
                continue

            video_path = os.path.join(root, file)
            feature_path = os.path.join(OUTPUT_DIR, file.replace(".mp4", ".npy"))

            if os.path.exists(feature_path):
                print(f"✅ Đã xử lý: {file}")
                continue  # Bỏ qua nếu đã trích xuất

            tensor = video_to_tensor(video_path)
            TC, H, W = tensor.shape
            tensor = tensor.reshape((TC // 3, 3, H, W)).permute(1, 0, 2, 3)
            if tensor is not None:
                batch_segments.append((tensor, feature_path))

            # Nếu đủ batch thì xử lý ngay
            if len(batch_segments) == BATCH_SIZE:
                process_batch(batch_segments)
                batch_segments.clear()  # Xóa dữ liệu sau khi xử lý

    # Xử lý nốt nếu còn segment lẻ
    if batch_segments:
        process_batch(batch_segments)

def process_batch(batch_segments):
    """ Trích xuất đặc trưng cho một batch và lưu kết quả """
    batch_tensors = torch.stack([item[0] for item in batch_segments])  # [B, 3, 32, 224, 224]
    features = extract_features(model, batch_tensors)  # [B, feature_dim]
    
    # Lưu đặc trưng từng file
    for i, (_, feature_path) in enumerate(batch_segments):
        np.save(feature_path, features[i])
        print(f"💾 Lưu: {feature_path}")

if __name__ == "__main__":
    process_segments()