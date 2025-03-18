#!/bin/bash

set -e  # Exit if a command fails

### 1️⃣ Install Dependencies (Aria2 & 7z) ###
install_aria2() {
    if ! command -v aria2c &> /dev/null; then
        echo "[INFO] Aria2 is not installed. Installing now..."
        sudo apt update && sudo apt install -y aria2
    fi
}

install_7z() {
    if ! command -v 7z &> /dev/null; then
        echo "[INFO] 7z is not installed. Installing now..."
        sudo apt update && sudo apt install -y p7zip-full
    fi
}

install_aria2
install_7z

### 2️⃣ Define Dataset Info ###
export SRC="https://Datasets" #  "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main"
DATASET_FILES=(
    "ipd_base.zip"
    "ipd_models.zip"
    "ipd_val.zip"
    "ipd_test_all.zip"
    "ipd_test_all.z01"
    # "ipd_train_pbr.zip"
    # "ipd_train_pbr.z01"
    # "ipd_train_pbr.z02"
    # "ipd_train_pbr.z03"
)

TARGET_DIR="./datasets"

# Ensure dataset directory exists
mkdir -p "$TARGET_DIR"

### 3️⃣ Fast Download Function (Aria2) ###
download_file() {
    local file="$1"
    local url="$SRC/$file"
    local filepath="$TARGET_DIR/$file"

    if [ -f "$filepath" ]; then
        echo "[INFO] File $filepath already exists. Skipping download."
        return 0
    fi

    echo "[INFO] Downloading $file with Aria2..."
    aria2c -x 16 -s 16 -k 1M -d "$TARGET_DIR" -o "$file" "$url"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Download failed for $file. Check your connection."
        exit 1
    fi
}

# Download all dataset files
for file in "${DATASET_FILES[@]}"; do
    download_file "$file"
done

### 4️⃣ Extract Files Exactly Like Original Commands ###
cd "$TARGET_DIR"

echo "[INFO] Extracting dataset as per original instructions..."

7z x ipd_base.zip             # Contains folder "ipd".
7z x ipd_models.zip -oipd     # Unpacks to "ipd".
7z x ipd_val.zip -oipd        # Unpacks to "ipd".
# 7z x ipd_test_all.zip -oipd   # Unpacks to "ipd".
# 7z x ipd_train_pbr.zip -oipd  # Unpacks to "ipd".

### 5️⃣ Cleanup (Remove Zip Files After Successful Extraction) ###
rm -f ipd_base.zip ipd_models.zip ipd_val.zip
# rm -f ipd_test_all.zip ipd_test_all.z01
# rm -f ipd_train_pbr.zip ipd_train_pbr.z01 ipd_train_pbr.z02 ipd_train_pbr.z03

echo "[INFO] Dataset fully downloaded, extracted, and cleaned up."
# #!/bin/bash

# # Define the URLs for the datasets
# DATASET_URLS=(
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_models.zip"
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_val.zip"
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_train_pbr.z01"
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_train_pbr.z02"
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_train_pbr.z03"
#     "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main/ipd_train_pbr.zip"
# )
# # docker run --network=host -e BOP_PATH=/datasets/ -e DATASET_NAME=ipd -v /home/akasha_uswest2/bin_picking_challenge/ipd_codebase/datasets/:/datasets/ -it ibpc:tester
# # Define the target directory
# TARGET_DIR="./datasets"

# # Create the target directory if it doesn't exist
# mkdir -p "$TARGET_DIR"

# # Function to download and extract a dataset
# download_and_extract() {
#     local url="$1"
#     local filename="$2"
    
#     echo "Downloading $filename..."
#     wget -O "$TARGET_DIR/$filename" "$url"
    
#     # Check if the download was successful
#     if [ $? -ne 0 ]; then
#         echo "Download failed for $filename. Please check your internet connection or the URL."
#         exit 1
#     fi
    
#     echo "Extracting $filename to $TARGET_DIR..."
#     unzip -q "$TARGET_DIR/$filename" -d "$TARGET_DIR"
    
#     # Remove the zip file after extraction
#     rm "$TARGET_DIR/$filename"
#     echo "Extraction complete for $filename."
# }

# # Loop through the dataset URLs and download/extract each one
# for url in "${DATASET_URLS[@]}"; do
#     filename=$(basename "$url")  # Extract the filename from the URL
#     download_and_extract "$url" "$filename"
# done

# echo "All downloads and extractions complete. Datasets are ready in $TARGET_DIR."
