#!/bin/bash

# Fail on error
set -e

# Directory where the script is executed is assumed to be the project root
DATA_DIR="./data"

echo "Setting up Cityscapes dataset in $DATA_DIR..."

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
fi

# Check if cityscapesScripts is installed
if ! command -v csDownload &> /dev/null
then
    echo "cityscapesScripts not found. Installing..."
    pip install cityscapesScripts
else
    echo "cityscapesScripts is already installed."
fi

# Check for credentials
if [ -z "$CITYSCAPES_USERNAME" ] || [ -z "$CITYSCAPES_PASSWORD" ]; then
    echo "----------------------------------------------------------------"
    echo "WARNING: CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD not set."
    echo "You may be prompted for credentials."
    echo "For automated download, export these variables before running."
    echo "----------------------------------------------------------------"
fi

# Enter data directory to download files there
cd "$DATA_DIR"

# Download the necessary packages
# leftImg8bit_trainvaltest.zip contains the images
# gtFine_trainvaltest.zip contains the ground truth
# echo "Downloading leftImg8bit_trainvaltest.zip..."
# csDownload leftImg8bit_trainvaltest.zip

echo "Downloading gtFine_trainvaltest.zip..."
csDownload gtFine_trainvaltest.zip

# Unzip the files
echo "Extracting leftImg8bit_trainvaltest.zip..."
unzip -q -o leftImg8bit_trainvaltest.zip

echo "Extracting gtFine_trainvaltest.zip..."
unzip -q -o gtFine_trainvaltest.zip

# Cleanup zip files (Optional - ask user if they want this?) 
# I will keep them for now to avoid re-downloading by mistake, or I could remove them.
# Usually saving space is preferred. I'll comment it out or leave it.
# Let's remove them to save space as the unzipped data is large.
echo "Cleaning up zip files..."
rm leftImg8bit_trainvaltest.zip
rm gtFine_trainvaltest.zip

echo "Cityscapes dataset setup complete."
