# Training Report for Human Action Recognition

## 📂 Dataset  
- **Source:** [Download Here](https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=NaN)

---

## 🛠 Training Configurations  

| Model Version | Description |
|--------------|-------------|
| **1.0** | Unforzen half of last block |
| **2.0** | Unfrozen last block |
| **3.0** | Unfrozen all |

### ⏳ Training Time on RTX 3090 (16GB VRAM)

| Batch Size | Time per Batch | Total Batches | Estimated Training Time per Epoch |
|------------|--------------|---------------|-------------------------|
| **45** | 2.58s | 287 | ~12m |

---

## 📊 Test Results (Version 3.0)  

| Metric | Value |
|--------|------|
| **Test Loss** | `0.4028` |
| **Test Accuracy** | `0.8359` |
| **Precision** | `0.6757` |
| **Recall** | `0.0375` |
| **F1 Score** | `0.0711` |

---

## 📁 Checkpoints and Logs  

| File | Description |
|------|-------------|
| `best_model.pth` | Best model parameters |
| `alllabel-props` | Precision-Recall curve data for validation set |
| `acc_loss.txt` | Logs of train/validation accuracy and loss |
| `result.txt` | Result on test set with optimal threshold |

### 📈 Log Format (`acc_loss.txt`)
Epoch TrainLoss TrainAcc ValLoss ValAcc ValPrecision ValRecall ValF1