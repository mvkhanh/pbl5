#!/bin/bash

# Clone repository
# echo "Cloning repository..."
# git clone https://github.com/mvkhanh/pbl5.git
# cd pbl5

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Downloading dataset..."
gdown -O UniformerData.zip "https://drive.google.com/uc?id=1AE0u2paq26metqutSJZaIfnXpOFy_QYE"

echo "Installing unzip and extracting dataset..."
sudo apt install unzip -y
unzip UniformerData.zip

echo "Reinstalling PyTorch..."
pip3 uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio

echo "Setup completed!"
