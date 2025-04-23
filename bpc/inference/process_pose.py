import os
import time
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

# Import your custom loss functions (if needed elsewhere)
from bpc.pose.models.losses import (
    EulerAnglePoseLoss,
    QuaternionPoseLoss,
    SixDPoseLoss,
    rotmat_from_euler,
    quat_to_rotmat,
    rotmat_from_6d,
)

from bpc.pose.models.simple_pose_net import SimplePoseNet
from bpc.utils.data_utils import letterbox_preserving_aspect_ratio, calc_pose_matrix
from bpc.inference.epipolar_matching import compute_cost_matrix, match_objects, triangulate_multi_view
from bpc.inference.yolo_detection import detect_with_yolo  # if used elsewhere
from bpc.inference.utils.camera_utils import load_camera_params, compute_fundamental_matrix

# -----------------------------------------------------------------------------
# Updated parameters class.
# If rotation_mode is None, the code will auto-detect it.
# -----------------------------------------------------------------------------
@dataclass
class PoseEstimatorParams:
    yolo_model_path: str = "yolo11-detection-obj11.pt"
    pose_model_path: str = "best_model.pth"
    matching_threshold: int = 30
    yolo_conf_thresh: float = 0.1
    rotation_mode: str = None  # if None, auto-detection will be used

# -----------------------------------------------------------------------------
# Function to load the pose model.
# This version auto-detects the rotation mode if not explicitly provided.
# -----------------------------------------------------------------------------
def load_pose_model(pose_model_path, device='cuda:0', rotation_mode=None):
    # Load checkpoint first.
    checkpoint = torch.load(pose_model_path, map_location=device)
    
    # Inspect the fully connected layer's weight to determine output dimension.
    # Adjust the key name if your checkpoint uses a different naming.
    if "fc.weight" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'fc.weight'.")
    fc_weight = checkpoint["fc.weight"]
    output_dim = fc_weight.shape[0]  # The number of rows corresponds to the output dimension.
    
    # Auto-detect rotation mode if not provided.
    if rotation_mode is None:
        if output_dim == 3:
            detected_mode = "euler"
        elif output_dim == 4:
            detected_mode = "quat"
        elif output_dim == 6:
            detected_mode = "6d"
        else:
            raise ValueError(f"Unexpected output dimension: {output_dim}. Cannot determine rotation mode.")
        print(f"Auto-detected rotation mode from checkpoint: {detected_mode}")
        rotation_mode = detected_mode
    else:
        print(f"Using user-specified rotation mode: {rotation_mode}")
    
    # Instantiate the model with the determined rotation mode.
    pose_model = SimplePoseNet(loss_type=rotation_mode, pretrained=False)
    pose_model.load_state_dict(checkpoint)
    pose_model.to(device).eval()
    return pose_model, rotation_mode


# -----------------------------------------------------------------------------
# A class for holding detection + capture information.
class PosePrediction:
    def __init__(self, detections, capture):
        self.boxes = np.array([x['bbox'] for x in detections])
        self.centroids = np.array([x['bb_center'] for x in detections])
        self.capture = capture
        self.t = self.triangulate()
        
    def triangulate(self):
        proj_mats = []
        for idx in range(len(self.boxes)):
            K = self.capture.Ks[idx]
            RT = self.capture.RTs[idx]
            P = K @ RT[:3]  # Projection matrix P = K * [R|t]
            proj_mats.append(P)
        X = triangulate_multi_view(proj_mats, self.centroids)
        return X

# -----------------------------------------------------------------------------
# Main estimator class.
class PoseEstimator:
    def __init__(self, params: PoseEstimatorParams):
        self.params = params
        # Initialize YOLO.
        from ultralytics import YOLO
        self.yolo = YOLO(params.yolo_model_path).cuda()
        
        # Load pose model and determine rotation mode (auto-detect if not provided)
        self.pose_model, self.rotation_mode = load_pose_model(
            pose_model_path=params.pose_model_path,
            device='cuda:0',
            rotation_mode=params.rotation_mode
        )
        print(f"Using rotation mode: {self.rotation_mode}")

    def _detect(self, capture):
        """
        Run YOLO on each image in the capture.
        Returns a dictionary mapping camera indices to a list of detections.
        Each detection is a dictionary with keys "bbox" and "bb_center".
        """
        camera_predictions = {}
        for idx, image in enumerate(capture.images):
            print(f"Processing image shape: {image.shape}")
            results = self.yolo(image, imgsz=1280)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss  = results.boxes.cls.cpu().numpy()
            if len(results.boxes) == 0:
                camera_predictions[idx] = []
                continue
            # Keep only detections with class==0 and conf>=threshold.
            valid = (clss == 0) & (confs >= self.params.yolo_conf_thresh)
            boxes = boxes[valid]
            preds_cam = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                preds_cam.append({
                    'bbox': (x1, y1, x2, y2),
                    'bb_center': (cx, cy),
                })
            camera_predictions[idx] = preds_cam
        return camera_predictions

    def _match(self, capture, detections):
        """
        Use epipolar geometry to match detections from multiple cameras.
        Returns a list of PosePrediction instances.
        """
        predictions = []
        # Assuming three cameras.
        det_cam1 = detections[0]
        det_cam2 = detections[1]
        det_cam3 = detections[2]
        K1, K2, K3 = capture.Ks
        R1, R2, R3 = [x[:3, :3] for x in capture.RTs]
        t1, t2, t3 = [x[:3, 3] for x in capture.RTs]
        F12 = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
        F13 = compute_fundamental_matrix(K1, R1, t1, K3, R3, t3)
        F23 = compute_fundamental_matrix(K2, R2, t2, K3, R3, t3)

        if len(det_cam1) == 0 or len(det_cam2) == 0 or len(det_cam3) == 0:
            print("\nAt least one camera has zero detections => no matching.")
            return predictions

        cost_matrix = compute_cost_matrix(det_cam1, det_cam2, det_cam3, F12, F13, F23)
        print("\n--- Cost Matrix Stats ---")
        print(f"Shape: {cost_matrix.shape}")
        print(f"Min: {cost_matrix.min():.4f}, Max: {cost_matrix.max():.4f}, Mean: {cost_matrix.mean():.4f}")

        # Optional: print some random samples from the cost matrix.
        N, M, P = cost_matrix.shape
        sample_count = min(5, N * M * P)
        print("\nRandom samples from cost_matrix:")
        for _ in range(sample_count):
            i = np.random.randint(0, N)
            j = np.random.randint(0, M)
            k = np.random.randint(0, P)
            val = cost_matrix[i, j, k]
            print(f"  cost_matrix[{i},{j},{k}] = {val:.4f}")

        # Hungarian matching + threshold
        matches = match_objects(cost_matrix, threshold=self.params.matching_threshold)
        matches_sorted = sorted(matches, key=lambda t: cost_matrix[t[0], t[1], t[2]])

        for i, j, k in matches_sorted:
            dets = [det_cam1[i], det_cam2[j], det_cam3[k]]
            predictions.append(PosePrediction(dets, capture))
        return predictions

    def _estimate_rotation(self, predictions):
        """
        For each PosePrediction, crop the detection from each image,
        run the pose model, and convert the output into a rotation matrix.
        """
        for p in predictions:
            rotation_preds = []
            # Loop over each camera's detection for this prediction.
            for idx in range(len(p.centroids)):
                x1, y1, x2, y2 = p.boxes[idx]
                img = p.capture.images[idx]
                crop = img[y1:y2, x1:x2]
                # Use letterbox_preserving_aspect_ratio to get a square image.
                letter_img, scale, dx, dy = letterbox_preserving_aspect_ratio(
                    crop, target_size=256, fill_color=(255, 255, 255)
                )
                letter_img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB)
                tens = TF.to_tensor(letter_img_rgb)
                tens = TF.normalize(tens, [0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
                tens = tens.unsqueeze(0).to('cuda')
                with torch.no_grad():
                    raw_pred = self.pose_model(tens)[0].cpu().numpy()
                    
                    if self.rotation_mode == "euler":
                        # Assume raw_pred contains [Rx, Ry, Rz] in radians.
                        wrapped_pred = ((raw_pred + np.pi) % (2 * np.pi)) - np.pi
                        euler_tensor = torch.tensor(wrapped_pred, dtype=torch.float32).unsqueeze(0)
                        rot_mat = rotmat_from_euler(euler_tensor).squeeze(0).numpy()
                    elif self.rotation_mode == "quat":
                        # Assume raw_pred contains a quaternion [x, y, z, w].
                        quat_tensor = torch.tensor(raw_pred, dtype=torch.float32).unsqueeze(0)
                        eps = 1e-8
                        quat_tensor = quat_tensor / (quat_tensor.norm(dim=1, keepdim=True) + eps)
                        rot_mat = quat_to_rotmat(quat_tensor).squeeze(0).numpy()
                    elif self.rotation_mode == "6d":
                        # Assume raw_pred contains a 6D representation.
                        rep6d_tensor = torch.tensor(raw_pred, dtype=torch.float32).unsqueeze(0)
                        rot_mat = rotmat_from_6d(rep6d_tensor).squeeze(0).numpy()
                    else:
                        raise ValueError("Unsupported rotation mode.")
                    
                    # Combine the estimated rotation with the camera extrinsics.
                    # (Current pipeline applies: cam_R.T @ model_R)
                    final_rot = p.capture.RTs[idx][:3, :3].T @ rot_mat
                    rotation_preds.append(final_rot)
            p.rotation_preds = rotation_preds
            # For example, use the rotation from the third camera.
            p.final_rotation = rotation_preds[2]
            p.pose = calc_pose_matrix(p.final_rotation, p.t)
