from inference.match_pipeline import process_scene  # or wherever your match_pipeline is

if __name__ == "__main__":
    # Define test configuration
    scene_dir = "/home/exouser/Desktop/idp_codebase/datasets/ipd_bop_data/train_pbr/000000"
    cam_ids = ["cam1", "cam2", "cam3"]
    yolo_model_path = "/home/exouser/Desktop/idp_codebase/yolo/models/detection/obj_11/yolo11-detection-obj_11.pt"
    output_dir = "/home/exouser/Desktop/idp_codebase/output/"

    # Run the scene processing
    process_scene(scene_dir, cam_ids, yolo_model_path, output_dir)

    print(f"Test completed. Results saved in {output_dir}")
