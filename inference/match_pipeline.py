import os
import numpy as np
import pandas as pd
from bop_eval.camera_utils import load_camera_params, compute_fundamental_matrix
from bop_eval.epipolar_matching import compute_cost_matrix, match_objects
from bop_eval.yolo_detection import detect_with_yolo


def process_scene(scene_dir, cam_ids, yolo_model_path, output_dir, threshold=30):
    """
    Process a scene to compute matched bounding box centers and save results.

    Args:
        scene_dir (str): Path to the scene directory.
        cam_ids (list): List of camera IDs.
        yolo_model_path (str): Path to the YOLO model.
        output_dir (str): Path to save output results.
        threshold (float): Threshold for epipolar matching.
    """
    # Create the output directory structure
    scene_id = os.path.basename(scene_dir)  # e.g., "000000"
    obj_id = os.path.basename(yolo_model_path).split('_')[-1].split('.')[0]  # e.g., "obj_11"
    obj_output_dir = os.path.join(output_dir, scene_id, obj_id)
    os.makedirs(obj_output_dir, exist_ok=True)

    # Load camera parameters
    cam_params = load_camera_params(scene_dir, cam_ids)
    num_images = len(os.listdir(os.path.join(scene_dir, "rgb_cam1")))

    all_results = []

    for image_id in range(num_images):
        # Compute fundamental matrices
        F12 = compute_fundamental_matrix(
            cam_params["cam1"]["K"][image_id], cam_params["cam1"]["R"][image_id], cam_params["cam1"]["t"][image_id],
            cam_params["cam2"]["K"][image_id], cam_params["cam2"]["R"][image_id], cam_params["cam2"]["t"][image_id],
        )
        F13 = compute_fundamental_matrix(
            cam_params["cam1"]["K"][image_id], cam_params["cam1"]["R"][image_id], cam_params["cam1"]["t"][image_id],
            cam_params["cam3"]["K"][image_id], cam_params["cam3"]["R"][image_id], cam_params["cam3"]["t"][image_id],
        )
        F23 = compute_fundamental_matrix(
            cam_params["cam2"]["K"][image_id], cam_params["cam2"]["R"][image_id], cam_params["cam2"]["t"][image_id],
            cam_params["cam3"]["K"][image_id], cam_params["cam3"]["R"][image_id], cam_params["cam3"]["t"][image_id],
        )

        # Detect objects in images using YOLO
        detections = detect_with_yolo(scene_dir, cam_ids, image_id, yolo_model_path)
        det_cam1, det_cam2, det_cam3 = detections["cam1"], detections["cam2"], detections["cam3"]

        # Skip if any camera has no detections
        if not (det_cam1 and det_cam2 and det_cam3):
            continue

        # Compute cost matrix and match objects
        cost_matrix = compute_cost_matrix(det_cam1, det_cam2, det_cam3, F12, F13, F23)
        matches = match_objects(cost_matrix, threshold)

        # Prepare match results
        cx_cy_array = []
        for i, j, k in matches:
            cx1, cy1 = det_cam1[i]["bb_center"]
            cx2, cy2 = det_cam2[j]["bb_center"]
            cx3, cy3 = det_cam3[k]["bb_center"]

            bbx1 = det_cam1[i]["bbox"]
            bbx2 = det_cam2[j]["bbox"]
            bbx3 = det_cam3[k]["bbox"]

            cx_cy_array.append([
                cx1, cy1, bbx1,
                cx2, cy2, bbx2,
                cx3, cy3, bbx3
            ])
            all_results.append([
                image_id, cx1, cy1, str(bbx1),
                cx2, cy2, str(bbx2),
                cx3, cy3, str(bbx3)
            ])

        # Save results for individual image
        if cx_cy_array:
            image_csv_path = os.path.join(obj_output_dir, f"{image_id:06d}.csv")
            pd.DataFrame(
                cx_cy_array,
                columns=[
                    "cam1_cx", "cam1_cy", "cam1_bbox",
                    "cam2_cx", "cam2_cy", "cam2_bbox",
                    "cam3_cx", "cam3_cy", "cam3_bbox"
                ],
            ).to_csv(image_csv_path, index=False)

    # Save results for the entire scene
    if all_results:
        scene_csv_path = os.path.join(obj_output_dir, f"match_{scene_id}.csv")
        pd.DataFrame(
            all_results,
            columns=[
                "image_id",
                "cam1_cx", "cam1_cy", "cam1_bbox",
                "cam2_cx", "cam2_cy", "cam2_bbox",
                "cam3_cx", "cam3_cy", "cam3_bbox"
            ],
        ).to_csv(scene_csv_path, index=False)
        print(f"Saved scene results to {scene_csv_path}")


if __name__ == "__main__":
    # Example arguments
    scene_dir = "/path/to/scene"
    cam_ids = ["cam1", "cam2", "cam3"]
    yolo_model_path = "/path/to/yolo_model.pt"
    output_dir = "/path/to/output"

    process_scene(scene_dir, cam_ids, yolo_model_path, output_dir)
