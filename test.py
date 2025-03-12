from dataset import get_dataloader
import torch
import torch.nn as nn
from model import get_model
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BATCH_SIZE = 30

CHECKPOINT_PATH = "ckpt/best_model.pth"

test_normal_path = 'UniformerData/Test/NormalVideos/'
test_abnormal_path = 'UniformerData/Test/Abnormal/'

def eval(model, loss_fn, data_loader):
    """Đánh giá mô hình trên tập test và tính Precision, Recall, F1-score."""
    model.eval()
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.float16):  # ✅ Dùng float16
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)  # Forward pass
            
            # Tính loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Chuyển output thành nhãn dự đoán (0 hoặc 1)
            preds = (outputs > 0.5).float()
            
            # Lưu lại dự đoán và nhãn thật
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Chuyển về numpy array
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Tính các chỉ số đánh giá
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    correct += (all_preds == all_labels).sum().item()    
    acc = correct / len(all_preds)

    avg_loss = total_loss / len(data_loader)

    return avg_loss, acc, precision, recall, f1

def load_checkpoint(model, checkpoint_path):
        """Nạp lại trạng thái từ checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"🔄 Load successfully!")

if __name__ == '__main__':
    test_loader = get_dataloader(test_abnormal_path, test_normal_path, batch_size=BATCH_SIZE)
    # Model
    model = get_model().to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
         load_checkpoint(model, CHECKPOINT_PATH)

    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, test_acc, precision, recall, test_f1_score = eval(model, loss_fn, test_loader)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f} | Precision: {precision} | Recall: {recall} | F1 score: {test_f1_score}')