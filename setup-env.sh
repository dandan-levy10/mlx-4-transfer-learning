#!/bin/bash

# Stop script on error
set -e

echo "=== Installing Miniconda ==="

# Download and install Miniconda
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh

echo "=== Creating Conda Environment ==="
conda create --name mlx-transfer-learning-env python=3.9 -y
conda activate mlx-transfer-learning-env

echo "=== Installing Packages ==="
# Install dependencies
conda install -y tqdm
pip install numpy torch datasets wandb sentencepiece torch torchvision torchaudio transformers nltk matplotlib


