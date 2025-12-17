#!/bin/bash
set -e

echo "========================================"
echo "Pre-downloading model weights and data"
echo "========================================"

export TORCH_HOME="${HOME}/.cache/torch"
export HF_HOME="${HOME}/.cache/huggingface"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

echo "Cache directories created:"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  HF_HOME: $HF_HOME"
echo ""

source activate cos429-final-env 2>/dev/null || conda activate cos429-final-env

python3 << 'EOF'
import torch
from torchvision import models

print("ResNet50 pre-trained weights")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("success")
EOF

echo "Downloading segmentation_models_pytorch encoders."
python3 << 'EOF'
import segmentation_models_pytorch as smp

print("Downloading ResNet50 encoder from segmentation_models_pytorch...")
try:
    encoder = smp.encoders.get_encoder(
        name="resnet50",
        weights="imagenet"
    )
    print("success")
except Exception as e:
    print(f"error {e}")
EOF

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
    print("success")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"error {e}")
EOF

echo "Downloading done"