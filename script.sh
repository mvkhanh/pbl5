#!/bin/bash

BRANCHES=("2.0" "3.0" "4.0")

# Thư mục lưu kết quả trên server
SERVER_SAVE_DIR="/root/trained_models"

# Thư mục đích trên máy local
LOCAL_SAVE_DIR="~/trained_models_backup"

# Thông tin kết nối SSH (thay đổi theo server của bạn)
USER="root"
SERVER_IP="103.78.3.95"

# Lặp qua từng branch
for BRANCH in "${BRANCHES[@]}"; do
    echo "🚀 Switching to branch: $BRANCH"
    git checkout $BRANCH

    echo "📢 Training model on $BRANCH..."
    python3 train.py  # Hoặc script train của bạn

    echo "💾 Saving model to $SERVER_SAVE_DIR/$BRANCH"
    mkdir -p "$SERVER_SAVE_DIR/$BRANCH"
    mv ckpt/best_model.pth "$SERVER_SAVE_DIR/$BRANCH/"

    echo "📤 Syncing data to local..."
    rsync -avz "$SERVER_SAVE_DIR/$BRANCH" "$USER@$SERVER_IP:$LOCAL_SAVE_DIR/"

    echo "✅ Done training on $BRANCH"
done

echo "🎉 All models trained and synced successfully!"