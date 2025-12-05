# PSHuman API Setup

Photorealistic single-image 3D human reconstruction using PSHuman.

## Requirements

- NVIDIA GPU with 40GB+ VRAM (recommended)
- CUDA 12.4 compatible drivers
- Python 3.10+

## Quick Start

### One-Click Setup

```bash
chmod +x setup.sh
./setup.sh
```

### Usage

```bash
# Generate 3D human from image
python generate.py --input image.png --output output/

# Start API server
python api.py

# Test API
curl -X POST http://localhost:8002/generate/upload -F 'file=@image.png'
```

## API Endpoints

- `GET /health` - Health check
- `POST /generate/upload` - Generate from uploaded image
- `GET /download/{filename}` - Download generated files

## Citation

```bibtex
@article{li2024pshuman,
  title={PSHuman: Photorealistic Single-view Human Reconstruction using Cross-Scale Diffusion},
  author={Li, Peng and others},
  journal={arXiv preprint arXiv:2409.10141},
  year={2024}
}
```
