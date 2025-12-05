#!/bin/bash
# PSHuman One-Click Setup Script
# Photorealistic single-image 3D human reconstruction

set -e

echo "=============================================="
echo "PSHuman One-Click Setup"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU required (40GB+ VRAM recommended)."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Clone the official repository
echo ""
echo "[1/6] Cloning PSHuman repository..."
if [ -d "PSHuman" ]; then
    echo "Repository already exists, updating..."
    cd PSHuman && git pull && cd ..
else
    git clone --depth 1 https://github.com/pengHTYX/PSHuman.git
fi

cd PSHuman

# Install PyTorch with CUDA
echo ""
echo "[2/6] Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 -q

# Install Kaolin
echo ""
echo "[3/6] Installing Kaolin..."
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html -q

# Install PyTorch3D
echo ""
echo "[4/6] Installing PyTorch3D..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" -q

# Install core dependencies
echo ""
echo "[5/6] Installing dependencies..."
pip install numpy==1.26.4 rembg warp-lang opencv-python-headless mediapipe einops kornia \
    omegaconf fire accelerate xformers transformers diffusers huggingface-hub \
    trimesh gradio fastapi uvicorn tqdm -q

# Install nvdiffrast
echo ""
echo "[6/6] Installing nvdiffrast..."
pip install "git+https://github.com/NVlabs/nvdiffrast.git" -q

# Copy API files
echo ""
echo "Copying API files..."
cp ../api.py . 2>/dev/null || true
cp ../generate.py . 2>/dev/null || true

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Note: PSHuman requires 40GB+ VRAM for best results."
echo ""
echo "Usage:"
echo "  # Generate 3D human from image:"
echo "  python generate.py --input image.png --output output/"
echo ""
echo "  # Start API server:"
echo "  python api.py"
echo ""
