import os
import subprocess
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import shutil
from transforms import (
    GroupCenterCrop,
    ToTorchFormatTensor,
    GroupNormalize,
    Stack,
    GroupScale
)
from uniformer import uniformer_base
from huggingface_hub import hf_hub_download

# Cấu hình thiết bị
device = 'cuda'
model = uniformer_base()

# Load model
model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device).eval()

# Cấu hình input
crop_size = 224
scale_size = 256
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

# Transform
transform = T.Compose([
    GroupScale(scale_size),
    GroupCenterCrop(crop_size),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std) 
])

# Hàm trích xuất đặc trưng nhiều segment cùng lúc
def extract_features(model, batch_segments):
    """
    batch_segments: Tensor có shape [B, 3, T, H, W]
    """
    with torch.no_grad():
        batch_segments = batch_segments.to(device)
        features = model.forward_features(batch_segments)
        gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        features = gap(features).squeeze()  # [B, C, 1, 1, 1] -> [B, C]
    return features.cpu().numpy()

# Hàm xử lý toàn bộ một video
def process_video(model, video_path, output_path, fps_sampling=5, segment_length=32, batch_size=8):
    """
    - Trích xuất frames từ video
    - Chia thành các segment (32 frames)
    - Xử lý nhiều segment trong một batch
    - Lưu toàn bộ features của video vào một file duy nhất
    """
    temp_frame_dir = "temp_frames1"
    os.makedirs(temp_frame_dir, exist_ok=True)

    # Trích xuất frames bằng ffmpeg
    frame_pattern = os.path.join(temp_frame_dir, "frame_%04d.png")
    ffmpeg_cmd = f'ffmpeg -i "{video_path}" -vf "fps={fps_sampling}" "{frame_pattern}" -hide_banner -loglevel error'
    subprocess.run(ffmpeg_cmd, shell=True)

    # Đọc danh sách frames
    frames = sorted(os.listdir(temp_frame_dir))
    if len(frames) < segment_length:
        print(f"Bỏ qua video {video_path} vì có ít hơn {segment_length} frames.")
        return

    # Chia thành các segment
    num_segments = len(frames) // segment_length
    segment_list = [frames[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]

    # Loại bỏ 20% segment đầu và cuối nếu không phải video "Normal"
    class_name = os.path.basename(os.path.dirname(video_path))
    if class_name != "NormalVideos" and num_segments > 4:
        num_remove = int(0.2 * num_segments)
        segment_list = segment_list[num_remove:-num_remove]

    # Xử lý theo batch
    all_features = []
    batch_segments = []
    for i, segment in enumerate(segment_list):
        segment_paths = [os.path.join(temp_frame_dir, frame) for frame in segment]
        img_groups = [Image.open(frame) for frame in segment_paths]
        transform_groups = transform(img_groups)  # Shape: [TC, H, W]

        # Reshape thành [1, 3, T, H, W]
        TC, H, W = transform_groups.shape
        transform_groups = transform_groups.reshape((1, TC // 3, 3, H, W)).permute(0, 2, 1, 3, 4)
        batch_segments.append(transform_groups)

        # Nếu đủ batch_size, đưa vào model
        if len(batch_segments) == batch_size or i == len(segment_list) - 1:
            batch_tensor = torch.cat(batch_segments, dim=0)  # [B, 3, T, H, W]
            features = extract_features(model, batch_tensor)
            all_features.append(features)
            batch_segments = []  # Reset batch

    # Gộp toàn bộ features của video thành một mảng numpy duy nhất
    if all_features:
        all_features = np.concatenate(all_features, axis=0)
        np.save(output_path, all_features)

    # Dọn dẹp thư mục frames
    shutil.rmtree(temp_frame_dir)

# Xử lý toàn bộ dataset
input_root = "Train"
output_root = "Segments"
fps_sampling = 5
segment_length = 32
batch_size = 8

for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for video_file in os.listdir(class_path):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(class_path, video_file)
        output_path = os.path.join(output_class_path, f"{video_file}.npy")

        if os.path.exists(output_path):
            print(f"✅ Đã xử lý {video_file}, bỏ qua.")
            continue

        print(f"📹 Đang xử lý video: {video_file}...")
        process_video(model, video_path, output_path, fps_sampling, segment_length, batch_size)

print("✅ Done!")