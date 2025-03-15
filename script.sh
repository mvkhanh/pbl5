#!/bin/bash

BRANCHES=("main" "2.0" "3.0")

# Thư mục lưu kết quả trên server
SERVER_SAVE_DIR="/root/trained_models"

# Lặp qua từng branch
for BRANCH in "${BRANCHES[@]}"; do
    echo "🚀 Switching to branch: $BRANCH"
    
    # Kiểm tra branch tồn tại (cả local và remote)
    if ! git ls-remote --heads origin $BRANCH | grep $BRANCH > /dev/null; then
        echo "❌ Branch $BRANCH không tồn tại!"
        continue
    fi
    
    # Chuyển sang branch
    git checkout $BRANCH || { echo "❌ Lỗi khi chuyển branch!"; exit 1; }

    echo "📢 Training model on $BRANCH..."
    python3 train.py || { echo "❌ Lỗi khi train model!"; exit 1; }

    python3 optimal_threshold.py || { echo "❌ Lỗi khi tối ưu threshold!"; exit 1; }

    echo "💾 Saving model to $SERVER_SAVE_DIR/$BRANCH"
    mkdir -p "$SERVER_SAVE_DIR/$BRANCH"

    # Copy thư mục ckpt sang server
    cp -r ckpt/* "$SERVER_SAVE_DIR/$BRANCH/"

    echo "✅ Done training on $BRANCH"
done

echo "🎉 All models trained and synced successfully!"