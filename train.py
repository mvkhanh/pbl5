from dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from Trainer import Trainer
from model import get_model
import argparse
import os

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.005
BATCH_SIZE = 45
PATIENCE = 7

# LOG_DIR = "logs"
train_path = 'UniformerData/Train/'
val_path = 'UniformerData/Validation/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='Choose 1, 2, 3', default=1)
    args = parser.parse_args()
    version = int(args.version)
    train_loader = get_dataloader(train_path, batch_size=BATCH_SIZE, isTrain=True)
    val_loader = get_dataloader(val_path, BATCH_SIZE)
    # Model
    model = get_model(version).to(DEVICE)
    CHECKPOINT_PATH = os.path.join(f"model{version}/best_model.pth")
    ACC_LOSS_PATH = os.path.join(f"model{version}/acc_loss.txt")
    os.makedirs(f'model{version}', exist_ok=True)
    # Train
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, train_loader, val_loader, loss_fn,
                  optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), epochs=EPOCHS, device=DEVICE, 
                  patience=PATIENCE, checkpoint_path=CHECKPOINT_PATH, acc_loss_path=ACC_LOSS_PATH)
    trainer.train()