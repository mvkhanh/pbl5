import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load dữ liệu từ server
data = np.load('precision_recall_data.npz')
labels = data['labels']
outputs = data['outputs']

# Tính toán đường Precision-Recall
precision, recall, _ = precision_recall_curve(labels, outputs)

# Vẽ biểu đồ
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()