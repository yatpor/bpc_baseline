# Object Detection and Pose Estimation Pipeline

## Overview
This repository provides an end-to-end pipeline for:
- Preparing data for YOLO-based object detection using Ultralytics’ YOLO.
- Training a YOLO model to detect a specific object.
- Training a simple pose estimation model (SimplePoseNet) on BOP-format data.

## Repository Structure
```
yolo/
  prepare_data.py   # Prepare data in YOLO format
  train.py          # Train YOLO using Ultralytics
  configs/         # YOLO configuration files (.yaml)
  models/          # YOLO trained weights (.pt)

pose/
  train.py         # Train the SimplePoseNet on BOP data
  checkpoints/     # Stores pose model checkpoints
  models/          # Network definitions, losses, etc.
  trainers/        # Training logic for pose estimation

utils/
  data_utils.py    # BOP dataset loading + transforms
  obj_match.py     # Example usage of matching pipeline

inference/
  # Scripts for epipolar matching, YOLO detection, and match pipelines

datasets/          # Contains BOP and YOLO data
runs/              # YOLO training runs, logs, etc.
output/            # Various output images and results
```

## Environment Setup
To set up the environment, follow these steps (tested on Ubuntu with an NVIDIA GPU). The environment name is `bop`.

### Create a Conda Environment
```bash
conda create -n bop python=3.10 -y
conda activate bop
```

### Install PyTorch and Dependencies
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Install CUDA Toolkit (Optional)
```bash
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
```

### Install Required Python Packages
```bash
pip install -r requirements.txt
```

### Check GPU Availability
```bash
nvidia-smi
nvcc --version
```

## Training Pipeline

### Prepare YOLO Data
Convert BOP data to YOLO format:
```bash
python yolo/prepare_data.py \
    --dataset_path "datasets/ipd_bop_data_jan25_1/train_pbr" \
    --output_path "datasets/yolo11/ipd_bop_data_jan25_1_obj_11" \
    --obj_id 11
```

### Train YOLO Model
```bash
python yolo/train.py \
    --obj_id 11 \
    --data_path "yolo/configs/data_obj_11.yaml" \
    --epochs 20 \
    --imgsz 640 \
    --batch 16 \
    --task detection
```

### Train Pose Model
```bash
python pose/train.py \
  --root_dir datasets/ipd_bop_data_jan25_1 \
  --target_obj_id 11 \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-3 \
  --num_workers 16 \
  --checkpoints_dir /home/exouser/Desktop/idp_codebase/pose/checkpoints
```

### Notes
- Ensure CUDA 12.1 drivers are installed and PyTorch recognizes the GPU (`nvidia-smi`).
- BOP dataset must follow standard conventions (train_pbr, test, etc.).
- Update `yolo/configs/data_obj_11.yaml` with the correct dataset paths.
- If encountering module import errors, try:
  ```bash
  python -m idp_codebase.pose.train ...
  ```
  or add `__init__.py` files where necessary.
