import re
from utils.transforms import (
    GroupScale,
    GroupCenterCrop,
    ToTorchFormatTensor,
    GroupNormalize,
    Stack
)
import os
import torchvision.transforms as T
import cv2
from PIL import Image

def extract_class_name(s):
    match = re.match(r'^[A-Za-z]+', s)
    return match.group(0)

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
    
    if len(frames) < 32:
        frames = frames + [frames[-1]] * (32 - len(frames))
    elif len(frames) > 32:
        frames = frames[:32]
    
    # Áp dụng transform
    frames = transform(frames).squeeze() # [3, 32, 224, 224]
    TC, H, W = frames.shape
    frames = frames.reshape((TC // 3, 3, H, W)).permute(1, 0, 2, 3)
    return frames

def get_all_videopaths(path):
    paths = []
    for root, _, files in os.walk(path):
        if files and files[0].endswith('mp4'):
            l = [os.path.join(root, file) for file in files]
            paths.extend(l)
    return paths

