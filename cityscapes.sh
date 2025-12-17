#!/bin/bash
set -e

DATA_DIR="./data"

echo "Setting up Cityscapes dataset in $DATA_DIR..."

if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi

if ! command -v csDownload &> /dev/null
then
    echo "Installing cityscapesScripts..."
    pip install cityscapesScripts
else
    echo "Already installed."
fi

cd "$DATA_DIR"

# leftImg8bit_trainvaltest.zip contains the images
# gtFine_trainvaltest.zip contains the ground truth


echo "Downloading gtFine_trainvaltest.zip..."
csDownload gtFine_trainvaltest.zip

echo "Extracting leftImg8bit_trainvaltest.zip..."
unzip -q -o leftImg8bit_trainvaltest.zip

echo "Extracting gtFine_trainvaltest.zip..."
unzip -q -o gtFine_trainvaltest.zip

rm leftImg8bit_trainvaltest.zip
rm gtFine_trainvaltest.zip

echo "Cityscapes dataset setup complete."
