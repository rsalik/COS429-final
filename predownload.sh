#!/bin/bash
# predownload.sh - Pre-download model weights and dependencies on login node
# Run this on the Della login node BEFORE submitting your SLURM job
# Usage: ./predownload.sh

set -e  # Exit on error

echo "========================================"
echo "Pre-downloading model weights and data"
echo "========================================"

# Set up cache directories
export TORCH_HOME="${HOME}/.cache/torch"
export HF_HOME="${HOME}/.cache/huggingface"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

echo "Cache directories created:"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  HF_HOME: $HF_HOME"
echo ""

# Activate your conda environment
echo "Activating conda environment..."
source activate cos429-final-env 2>/dev/null || conda activate cos429-final-env

echo ""
echo "Downloading PyTorch ResNet50 weights..."
python3 << 'EOF'
import torch
from torchvision import models

print("Downloading ResNet50 pre-trained weights...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("✓ ResNet50 weights downloaded successfully")
EOF

echo ""
echo "Downloading segmentation_models_pytorch encoders..."
python3 << 'EOF'
import segmentation_models_pytorch as smp

print("Downloading ResNet50 encoder from segmentation_models_pytorch...")
try:
    encoder = smp.encoders.get_encoder(
        name="resnet50",
        weights="imagenet"
    )
    print("✓ ResNet50 encoder downloaded successfully")
except Exception as e:
    print(f"Warning: Could not pre-download via segmentation_models_pytorch: {e}")
    print("This is OK - will fall back to PyTorch weights on compute node")
EOF

echo ""
echo "Testing model initialization (without training)..."
python3 << 'EOF'
import torch
import segmentation_models_pytorch as smp

print("Creating U-Net model with ResNet50 backbone...")
try:
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=19,
        activation=None,
    )
    print("✓ U-Net model created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"Warning: Model creation failed: {e}")
    print("This may be due to network issues, but cached weights should work")
EOF

echo ""
echo "========================================"
echo "Pre-download complete!"
echo "========================================"
echo ""
echo "Cache contents:"
du -sh "$TORCH_HOME" 2>/dev/null || echo "  (TORCH_HOME not found yet)"
du -sh "$HF_HOME" 2>/dev/null || echo "  (HF_HOME not found yet)"
echo ""
echo "Next steps:"
echo "1. Update your job.slurm to include these environment variables:"
echo "   export TORCH_HOME=\${HOME}/.cache/torch"
echo "   export HF_HOME=\${HOME}/.cache/huggingface"
echo "2. Submit your job: sbatch job.slurm"
