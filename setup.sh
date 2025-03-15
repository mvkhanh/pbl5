#!/bin/bash

# Clone repository
# echo "Cloning repository..."
# git clone https://github.com/mvkhanh/pbl5.git
# cd pbl5

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo "Downloading dataset..."
# gdown -O UniformerData.zip "https://drive.google.com/uc?id=1EpM1kL4-OnLFB4ZHrF0i1luoqSYivAov"
gdown -O UniformerData.zip "https://drive.google.com/uc?id=11B3ddNl-ATcJLh0j0pBt1FNHvkwley0x"

echo "Installing unzip and extracting dataset..."
sudo apt install unzip -y
unzip UniformerData.zip

echo "Reinstalling PyTorch..."
pip3 uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio

mv UniformerData2/ UniformerData

echo "Setup completed!"

