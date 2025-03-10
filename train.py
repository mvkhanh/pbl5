from dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from Trainer import Trainer

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.005
BATCH_SIZE = 32
PATIENCE = 7
CHECKPOINT_PATH = "ckpt/best_model.pth"
# LOG_DIR = "logs"
abnormal_path = 'UniformerData/Abnormal/'
normal_path = 'UniformerData/NormalVideos/'

if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(abnormal_path, normal_path, BATCH_SIZE)

    # Model
    model = get_model().to(DEVICE)

    # Train
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, train_loader, val_loader, loss_fn,
                  optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), epochs=EPOCHS, device=DEVICE, 
                  patience=PATIENCE, checkpoint_path=CHECKPOINT_PATH)
    trainer.train()