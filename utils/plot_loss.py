import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file txt
def read_data(file_path):
    data = np.loadtxt(file_path)
    epochs = data[:, 0]
    train_loss = data[:, 1]
    val_loss = data[:, 3]
    return epochs, train_loss, val_loss

def plot_loss(epochs, train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss Curve')
    plt.show()

if __name__ == '__main__':
    file_path = '../acc_loss1.txt'
    epochs, train_loss, val_loss = read_data(file_path)
    plot_loss(epochs, train_loss, val_loss)