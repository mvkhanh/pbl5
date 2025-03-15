import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device='cuda', epochs=50, patience=7, checkpoint_path="checkpoint.pth", acc_loss_path='acc_loss.txt'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.acc_loss_path = acc_loss_path

        self.criterion = loss_fn
        self.optimizer = optimizer
        self.threshold = 0.5
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.scaler = torch.amp.GradScaler('cuda')  # ✅ Thêm GradScaler cho AMP

        # 🟢 Kiểm tra checkpoint có tồn tại không
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if os.path.exists(checkpoint_path):
            self._load_checkpoint()

    def _save_checkpoint(self, epoch, val_loss):
        """Lưu trạng thái model, optimizer và epoch."""
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),  # ✅ Lưu cả trạng thái GradScaler
            'val_loss': val_loss
        }
        torch.save(state, self.checkpoint_path)
        print(f"✅ Checkpoint saved at epoch {epoch}")

    def _load_checkpoint(self):
        """Nạp lại trạng thái từ checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])  # ✅ Load lại GradScaler
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['val_loss']
        print(f"🔄 Resuming training from epoch {self.start_epoch}")

    def _eval(self):
        all_preds = []
        all_labels = []
        total_loss = 0
        self.model.eval()
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16):  # ✅ Dùng float16
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)  # Forward pass
                
                # Tính loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = torch.sigmoid(outputs)
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
        acc = (all_preds == all_labels).mean() 

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, acc, precision, recall, f1

    def train(self):
        """Chạy quá trình huấn luyện."""
        early_stopping_counter = 0

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                self.train_loader.dataset.dataset._sample_normal()
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.autocast(device_type=self.device, dtype=torch.float16):  # ✅ Bật AMP
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()  # ✅ Dùng scaler để backward
                    self.scaler.step(self.optimizer)
                    self.scaler.update()  # ✅ Cập nhật scale
                    outputs = torch.sigmoid(outputs)
                    total_loss += loss.item()
                    preds = (outputs > self.threshold).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    pbar.set_postfix(loss=loss.item(), acc=correct/total)
                    pbar.update(1)

            train_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            val_loss, val_acc, val_precision, val_recall, val_f1 = self._eval()

            with open(self.acc_loss_path, 'a') as f:
                f.write(f'{epoch} {train_loss:.4f} {train_acc:.4f} {val_loss:.4f} {val_acc:.4f} {val_precision:.4f} {val_recall:.4f} {val_f1:.4f}\n')

            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val Precision {val_precision:.4f} | Val Recall {val_recall:.4f} | Val F1 {val_f1:.4f}")

            self.scheduler.step(val_loss)

            # 💾 Lưu checkpoint nếu tốt nhất
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # ⛔ Early stopping
            if early_stopping_counter >= self.patience:
                print("⛔ Early stopping triggered!")
                break