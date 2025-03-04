import os
import requests
import zipfile
import shutil

# Định nghĩa đường dẫn thư mục
download_list_file = "download.txt"
download_dir = "downloads"
extract_dir = "temp_extracted"
output_dir = "Train/NormalVideos"

# Tạo thư mục nếu chưa có
os.makedirs(download_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Đọc danh sách URL từ file download.txt
with open(download_list_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    file_name = os.path.join(download_dir, url.split("/")[-1].split("?")[0])  # Lấy tên file zip
    print(f"⬇️ Đang tải {file_name}...")

    # Tải file zip từ Dropbox
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"✅ Đã tải xong {file_name}")
    else:
        print(f"❌ Lỗi khi tải {file_name}")
        continue

    # Giải nén file zip
    os.makedirs(extract_dir, exist_ok=True)
    print(f"📂 Giải nén {file_name}...")
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Di chuyển tất cả video vào Train/NormalVideos
    print("🚀 Di chuyển video vào Train/NormalVideos...")
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):  # Thêm định dạng video nếu cần
                src_path = os.path.join(root, file)
                dest_path = os.path.join(output_dir, file)
                shutil.move(src_path, dest_path)

    # Xóa file zip và thư mục tạm
    os.remove(file_name)
    shutil.rmtree(extract_dir)

print("🎉 Hoàn thành! Tất cả video đã được gom vào Train/NormalVideos.")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transforms import (
    GroupCenterCrop,
    ToTorchFormatTensor,
    GroupNormalize,
    Stack,
    GroupScale
)
from uniformer import uniformer_base
from huggingface_hub import hf_hub_download

device = 'cuda'
model = uniformer_base()
# load state uniformer_base_k600_32x4.pth  uniformer_small_k600_16x8.pth
model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
# set to eval mode
model = model.to(device)
model = model.eval()

# Cấu hình input
crop_size = 224
scale_size = 256
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

# Khởi tạo transform
transform = T.Compose([
    GroupScale(scale_size),
    GroupCenterCrop(crop_size),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std) 
])

# Hàm trích xuất đặc trưng
def extract_features(model, video_clip):
    with torch.no_grad():
        video_clip = video_clip.to(device)
        features = model.forward_features(video_clip)  # Output có thể là [B, C, T, H, W]

        # Global Average Pooling (GAP)
        gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        features = gap(features).squeeze()  # [B, C, 1, 1, 1] -> [B, C]
    return features

def get_features_from_segment(model, segment):
    # Chuyển frame thành list ảnh PIL và áp dụng transform
    img_groups = [Image.open(frame) for frame in segment]
    transform_groups = transform(img_groups)  # Shape: [TC, H, W]

    # Reshape về (B, 3, 32, H, W)
    TC, H, W = transform_groups.shape
    transform_groups = transform_groups.reshape((1, TC // 3, 3, H, W)).permute(0, 2, 1, 3, 4)
    features = extract_features(model, transform_groups)
    return features


import os
import subprocess
import numpy as np
import cv2

# Thư mục dữ liệu gốc
input_root = "Train"
output_root = "Segments"

# Số frames mỗi giây cần trích xuất
fps_sampling = 5  # Mỗi giây lấy 5 frames
segment_length = 32  # Mỗi segment có 32 frames

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
        segment_prefix = os.path.join(output_class_path, f"{video_file}_segment_")

        # Kiểm tra xem đã xử lý bao nhiêu segments
        existing_segments = sorted([
            int(fname.split("_")[-1].split(".")[0])  # Lấy số thứ tự segment
            for fname in os.listdir(output_class_path)
            if fname.startswith(video_file) and fname.endswith(".npy")
        ])

        last_processed_segment = existing_segments[-1] if existing_segments else -1
        print(f"🔄 Tiếp tục xử lý {video_file} từ segment {last_processed_segment + 1}...")

        temp_frame_dir = "temp_frames"
        os.makedirs(temp_frame_dir, exist_ok=True)

        if not existing_segments:
            # Nếu chưa có segment nào, trích xuất lại frames
            frame_pattern = os.path.join(temp_frame_dir, "frame_%04d.png")
            ffmpeg_cmd = f'ffmpeg -i "{video_path}" -vf "fps={fps_sampling}" "{frame_pattern}" -hide_banner -loglevel error'
            subprocess.run(ffmpeg_cmd, shell=True)

        frames = sorted(os.listdir(temp_frame_dir))
        num_segments = len(frames) // segment_length
        segment_list = [frames[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]

        if class_name != "NormalVideos":
            num_remove = int(0.2 * num_segments)
            segment_list = segment_list[num_remove:-num_remove]

        for i, segment in enumerate(segment_list):
            if i <= last_processed_segment:
                continue  # Bỏ qua các segment đã lưu

            segment_output_path = os.path.join(output_class_path, f"{video_file}_segment_{i:03d}.npy")
            segment = [os.path.join(temp_frame_dir, frame) for frame in segment]
            features = get_features_from_segment(model, segment)
            np.save(segment_output_path, np.array(features))

        # Xóa frames sau khi xử lý xong
        for frame in os.listdir(temp_frame_dir):
            os.remove(os.path.join(temp_frame_dir, frame))
        os.rmdir(temp_frame_dir)

print("✅ Done! Tiếp tục xử lý xong tất cả các video bị gián đoạn.")