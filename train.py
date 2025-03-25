from dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from Trainer import Trainer
from model import get_model
import argparse

# ---------------------- Cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.005
BATCH_SIZE = 1
PATIENCE = 7
CHECKPOINT_PATH = "ckpt/best_model.pth"
# LOG_DIR = "logs"
train_path = 'UniformerData/Train/'
val_path = 'UniformerData/Validation/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='Choose 1, 2, 3')
    args = parser.parse_args()

    train_loader = get_dataloader(train_path, batch_size=BATCH_SIZE, isTrain=True)
    val_loader = get_dataloader(val_path, BATCH_SIZE)
    # Model
    model = get_model(int(args.version)).to(DEVICE)

    # Train
    loss_fn = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, train_loader, val_loader, loss_fn,
                  optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), epochs=EPOCHS, device=DEVICE, 
                  patience=PATIENCE, checkpoint_path=CHECKPOINT_PATH)
    trainer.train()