import os
import argparse
import numpy as np
import pandas as pd
import cv2
from bop_eval.camera_utils import load_camera_params, compute_fundamental_matrix
from bop_eval.triangulation import triangulate_multi_view, compute_final_pose
from data.bop_dataset import letterbox_preserving_aspect_ratio
from models.simple_pose_net import SimplePoseNet
import torch


def process_pose_estimation(scene_dir, cam_ids, csv_path, output_dir, model_path):
    """
    Process pose estimation using matched data from CSV and a trained model.

    Args:
        scene_dir (str): Path to the scene directory.
        cam_ids (list): List of camera IDs.
        csv_path (str): Path to the CSV file with matched bounding box data.
        output_dir (str): Path to save output results.
        model_path (str): Path to the pretrained rotation prediction model.
    """
    # Load camera parameters
    cam_params = load_camera_params(scene_dir, cam_ids)

    # Load matched bounding box centers
    matched_data = pd.read_csv(csv_path)
    num_objects = matched_data.shape[0]

    # Initialize triangulated points and rotation results
    triangulated_points = []
    final_pose_array = np.zeros((num_objects, len(cam_ids), 5), dtype=np.float32)

    # Load pretrained PoseNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Process each matched group
    for i, row in matched_data.iterrows():
        # Extract bounding box centers and bounding boxes
        points_2d = []
        crops = []
        for cam_id in cam_ids:
            cx, cy = row[f"{cam_id}_cx"], row[f"{cam_id}_cy"]
            bbox = eval(row[f"{cam_id}_bbox"])  # Convert string to tuple
            points_2d.append((cx, cy))

            # Load image and crop bounding box
            img_path = os.path.join(scene_dir, f"rgb_{cam_id}", f"{int(row['image_id']):06d}.jpg")
            img = cv2.imread(img_path)
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            crop, _, _, _ = letterbox_preserving_aspect_ratio(crop)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(torch.tensor(crop.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device))

        # Perform triangulation
        proj_mats = []
        for cam_id in cam_ids:
            K = cam_params[cam_id]['K'][int(row['image_id'])]
            R = cam_params[cam_id]['R'][int(row['image_id'])]
            t = cam_params[cam_id]['t'][int(row['image_id'])]
            P = K @ np.hstack([R, t.reshape(-1, 1)])
            proj_mats.append(P)

        X = triangulate_multi_view(proj_mats, points_2d)
        triangulated_points.append(X)

        # Predict rotations using the model
        with torch.no_grad():
            preds = [model(crop).cpu().numpy().flatten() for crop in crops]

        # Store predictions: [Rx, Ry, Rz, cx, cy]
        for cam_idx, pred in enumerate(preds):
            final_pose_array[i, cam_idx, :3] = pred[:3]
            final_pose_array[i, cam_idx, 3:] = points_2d[cam_idx]

    # Combine final rotations and translations
    triangulated_points = np.array(triangulated_points, dtype=np.float32)
    final_6d_pose = compute_final_pose(final_pose_array, triangulated_points)

    # Save final 6D poses
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_6d_pose.csv")
    pd.DataFrame(
        final_6d_pose,
        columns=["Rx", "Ry", "Rz", "X", "Y", "Z"]
    ).to_csv(output_file, index=False)
    print(f"Final 6D poses saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pose estimation.")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to the scene directory.")
    parser.add_argument("--cam_ids", nargs="+", default=["cam1", "cam2", "cam3"], help="List of camera IDs.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the matched bounding box CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")

    args = parser.parse_args()

    process_pose_estimation(
        scene_dir=args.scene_dir,
        cam_ids=args.cam_ids,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
