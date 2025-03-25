#!/bin/bash

# Thư mục lưu kết quả trên server
SERVER_SAVE_DIR="/root/trained_models"

echo "📢 Training model..."
python3 train.py || { echo "❌ Lỗi khi train model!"; exit 1; }

python3 optimal_threshold.py || { echo "❌ Lỗi khi tối ưu threshold!"; exit 1; }

echo "💾 Saving model to $SERVER_SAVE_DIR"
mkdir -p "$SERVER_SAVE_DIR"

# Copy thư mục ckpt sang server
cp -r ckpt/* "$SERVER_SAVE_DIR"

echo "✅ Done training"

echo "🎉 All models trained and synced successfully!"