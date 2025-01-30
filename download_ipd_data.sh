#!/bin/bash

# Define the URLs for the datasets
DATASET_URLS=(
    "https://storage.googleapis.com/akasha-public/IBPC/ipd_27.01.zip"
    "https://storage.googleapis.com/akasha-public/IBPC/bpc_train_0.zip"
    "https://storage.googleapis.com/akasha-public/IBPC/bpc_train_1.zip"
)

# Define the target directory
TARGET_DIR="./datasets"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Function to download and extract a dataset
download_and_extract() {
    local url="$1"
    local filename="$2"
    
    echo "Downloading $filename..."
    wget -O "$TARGET_DIR/$filename" "$url"
    
    # Check if the download was successful
    if [ $? -ne 0 ]; then
        echo "Download failed for $filename. Please check your internet connection or the URL."
        exit 1
    fi
    
    echo "Extracting $filename to $TARGET_DIR..."
    unzip -q "$TARGET_DIR/$filename" -d "$TARGET_DIR"
    
    # Remove the zip file after extraction
    rm "$TARGET_DIR/$filename"
    echo "Extraction complete for $filename."
}

# Loop through the dataset URLs and download/extract each one
for url in "${DATASET_URLS[@]}"; do
    filename=$(basename "$url")  # Extract the filename from the URL
    download_and_extract "$url" "$filename"
done

echo "All downloads and extractions complete. Datasets are ready in $TARGET_DIR."