from dataset import get_dataloader
import torch
import torch.nn as nn
from model import get_model
import os
import numpy as np
from utils.utils import video_to_tensor
import cv2
from time import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
CHECKPOINT_PATH = "ckpt/best_model2.pth"
THRESHOLD = 0.4856

test_path = 'UniformerData/Test'

def load_checkpoint(model, checkpoint_path):
        """Nạp lại trạng thái từ checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"🔄 Load successfully!")
#Abnormal/Explosion016_x264_5fps_0000.mp4
#Abnormal/Shooting032_x264_5fps_0042.mp4
if __name__ == '__main__':
    # Model
    model = get_model().to(DEVICE)
    path = 'UniformerData/Test/Abnormal/Explosion016_x264_5fps_0000.mp4'
    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(model, CHECKPOINT_PATH)

    model.eval()
    with torch.no_grad():
        video_path = os.path.join(path)
        tensor = video_to_tensor(video_path).unsqueeze(0)
        t1 = time()
        output = torch.sigmoid(model(tensor)).item()
        print(time() - t1)
        anomaly_text = f"Anomaly Probability: {output:.4f}"
        if output > THRESHOLD:
            anomaly_text += " | Pham phap!"
            color = (0, 0, 255)  # Đỏ
        else:
            anomaly_text += " | Binh thuong"
            color = (0, 255, 0)  # Xanh lá
        print(anomaly_text)

    vc = cv2.VideoCapture(video_path)
    FPS = 5
        # Hiển thị frame đầu tiên để đợi người dùng nhấn phím
    ret, frame = vc.read()
    if ret:
        scale_factor = 3.5
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

        cv2.putText(frame, "Press SPACE to start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Anomaly Detection", frame)

        # Đợi bấm phím SPACE (phím 32) mới bắt đầu chạy
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32:  # Phím Space
                break

        # Bắt đầu phát video
        while True:
            ret, frame = vc.read()
            if not ret:
                break

            frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

            cv2.putText(frame, anomaly_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, color, 2, cv2.LINE_AA)

            cv2.imshow("Anomaly Detection", frame)

            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
                break
