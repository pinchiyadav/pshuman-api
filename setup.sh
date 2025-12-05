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
echo "[1/7] Cloning PSHuman repository..."
if [ -d "PSHuman" ]; then
    echo "Repository already exists, updating..."
    cd PSHuman && git pull && cd ..
else
    git clone --depth 1 --progress https://github.com/pengHTYX/PSHuman.git
fi

cd PSHuman

# Create virtual environment to avoid conflicts
echo ""
echo "[2/7] Creating isolated virtual environment..."
python -m venv pshuman_env --system-site-packages 2>/dev/null || python3 -m venv pshuman_env --system-site-packages
source pshuman_env/bin/activate

# Install PyTorch with CUDA
echo ""
echo "[3/7] Installing PyTorch with CUDA support..."
pip install --progress-bar on torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install specific versions to avoid conflicts
echo ""
echo "[4/7] Installing core dependencies with pinned versions..."
pip install --progress-bar on \
    "diffusers==0.21.4" \
    "transformers==4.35.2" \
    "huggingface-hub==0.19.4" \
    "accelerate==0.25.0" \
    numpy==1.26.4 einops kornia omegaconf fire tqdm \
    opencv-python-headless mediapipe rembg trimesh \
    gradio fastapi uvicorn

# Install Kaolin
echo ""
echo "[5/7] Installing Kaolin..."
pip install --progress-bar on kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html || echo "Kaolin install skipped"

# Install PyTorch3D
echo ""
echo "[6/7] Installing PyTorch3D..."
pip install --progress-bar on "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5" || echo "PyTorch3D install skipped"

# Install nvdiffrast
echo ""
echo "[7/7] Installing nvdiffrast..."
pip install --progress-bar on "git+https://github.com/NVlabs/nvdiffrast.git" || echo "nvdiffrast install skipped"

# Copy API files
echo ""
echo "Copying API files..."
cd ..
cp api.py PSHuman/ 2>/dev/null || true
cp generate.py PSHuman/ 2>/dev/null || true
cd PSHuman

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
source pshuman_env/bin/activate
echo "PSHuman environment activated"
EOF
chmod +x activate.sh

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Note: PSHuman requires 40GB+ VRAM for best results."
echo ""
echo "Usage:"
echo "  # Activate environment first:"
echo "  source PSHuman/activate.sh"
echo ""
echo "  # Generate 3D human from image:"
echo "  python inference.py --config configs/inference-768-6view.yaml \\"
echo "    pretrained_model_name_or_path='pengHTYX/PSHuman_Unclip_768_6views' \\"
echo "    validation_dataset.root_dir='examples' \\"
echo "    num_views=7 save_mode='rgb'"
echo ""
echo "  # Start API server:"
echo "  python api.py"
echo ""
