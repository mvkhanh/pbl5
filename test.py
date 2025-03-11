from dataset import get_dataloader
import torch
import torch.nn as nn
from model import get_model
import os

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BATCH_SIZE = 32

CHECKPOINT_PATH = "ckpt/best_model.pth"

test_normal_path = 'UniformerData/Test/NormalVideos/'
test_abnormal_path = 'UniformerData/Test/Abnormal/'

def eval(model, loss_fn, data_loader):
        """Đánh giá mô hình trên tập validation."""
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.inference_mode():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / len(data_loader)
        val_acc = correct / total
        return val_loss, val_acc

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
    test_loss, test_acc = eval(model, loss_fn, test_loader)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}')