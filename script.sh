#!/bin/bash
set -e
# Thư mục lưu kết quả trên server
SERVER_SAVE_DIR="/root/trained_models"

for i in {1..3}
do
    echo "📢 Training model..."
    python3 train.py $i || { echo "❌ Lỗi khi train model $i!"; exit 1; }

    python3 optimal_threshold.py $i || { echo "❌ Lỗi khi tối ưu threshold model $i!"; exit 1; }

    echo "💾 Saving model to $SERVER_SAVE_DIR/model$i"
    mkdir -p "$SERVER_SAVE_DIR/model$i"

    # Copy thư mục ckpt sang server
    cp -r ckpt/* "$SERVER_SAVE_DIR/model$i"

    echo "✅ Done training"
done

echo "🎉 All models trained and synced successfully!"