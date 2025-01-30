# Datasets Folder Structure

This folder contains various datasets used for object detection and pose estimation.

## Structure
```
datasets/
├── idp/  # Test folder
├── train_folder/  # Training folder
│   ├── models_eval/
│   │   ├── obj_000000.ply
│   │   ├── obj_000001.ply
│   │   ├── ...
│   │   ├── obj_000020.ply
│   ├── train_pbr/
│   │   ├── 000000/
│   │   │   ├── aolp_cam1/
│   │   │   ├── depth_cam1/
│   │   │   ├── dolp_cam1/
│   │   │   ├── mask_cam1/
│   │   │   ├── mask_visib_cam1/
│   │   │   ├── rgb_cam1/
│   │   │   ├── scene_camera_cam1.json
│   │   │   ├── scene_gt_cam1.json
│   │   │   ├── scene_gt_info_cam1.json
│   │   ├── 000001/
│   │   ├── 000002/
│   │   ├── 000003/
│   │   ├── 000004/
├── yolo11/
│   ├── ipd_bop_data_jan25_1_obj_11/
│   │   ├── images/
│   │   ├── labels/
```

## Description
- `idp/` is the test dataset folder.
- `train_folder/` stores processed datasets with subdirectories:
  - `models_eval/` contains `.ply` files representing 3D object models.
  - `train_pbr/` holds multiple scenes, each containing different camera captures and metadata JSON files.
- `yolo11/` contains YOLO-formatted datasets:
  - `images/` holds training images.
  - `labels/` stores corresponding YOLO labels.

### YOLO11 Folder Structure
After running `yolo/prepare_data.py`, the `yolo11/` folder should be structured as shown above.

## Notes
- Ensure correct paths are set in scripts when accessing dataset files.
- Structure follows the BOP dataset conventions and Ultralytics YOLO format.
