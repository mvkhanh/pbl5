from dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from mymodel import MyModel
from Trainer import Trainer
from model import get_model

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.005
BATCH_SIZE = 32
PATIENCE = 5
CHECKPOINT_PATH = "ckpt/best_model.pth"
# LOG_DIR = "logs"
train_abnormal_path = 'UniformerData/Train/Abnormal/'
train_normal_path = 'UniformerData/Train/NormalVideos/'
test_normal_path = 'UniformerData/Test/NormalVideos/'
test_abnormal_path = 'UniformerData/Test/Abnormal/'

if __name__ == '__main__':
    # train_loader = get_dataloader(train_abnormal_path, train_normal_path, batch_size=BATCH_SIZE, isTrain=True)
    # test_loader = get_dataloader(test_abnormal_path, test_normal_path, batch_size=BATCH_SIZE)
    train_loader, test_loader = get_dataloader(train_abnormal_path, train_normal_path, batch_size=BATCH_SIZE, split_size=0.15)
    # Model
    model = get_model().to(DEVICE)

    # Train
    pos_weight = torch.tensor([2.5])  # Tăng trọng số lớp dương
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    trainer = Trainer(model, train_loader, test_loader, loss_fn,
                  optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), epochs=EPOCHS, device=DEVICE, 
                  patience=PATIENCE, checkpoint_path=CHECKPOINT_PATH)
    trainer.train()