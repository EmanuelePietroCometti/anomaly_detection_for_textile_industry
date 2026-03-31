from pathlib import Path
import shutil
import os
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import cv2
import numpy as np

def backup_and_cleanup_latest_run(symlink_path, dest_parent_dir, backbone, layers, config):
    """
    Resolves a symlink copies its target directory to a new
    custom-named folder, and safely deletes both the original target and the symlink.
    """
    dataset_cfg = config["dataset_pipeline"]
    dataset_version = dataset_cfg["dataset_version"]
    if not os.path.islink(symlink_path):
        print(f"[ERROR] '{symlink_path}' is not a valid symbolic link.")
        return

    real_source_dir = Path(symlink_path)
    print(f"Symlink resolved. Actual source directory: {real_source_dir}")

    timestamp = config["global_timestamp"]
    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    new_folder_name = f"{timestamp}_{backbone}_{layers_str}_d{dataset_version}"
    destination_dir = Path(dest_parent_dir) / new_folder_name

    try:
        print(f"Copying files to: {destination_dir}")
        shutil.copytree(real_source_dir, destination_dir, dirs_exist_ok=True)
        print("[SUCCESS] Backup completed successfully.")
        print(f"Deleting original directory: {real_source_dir}")
        shutil.rmtree(real_source_dir)
        print(f"Removing symlink: {symlink_path}")
        os.unlink(symlink_path)
        print("[SUCCESS] Cleanup completed. Only the customized backup remains.")

    except Exception as e:
        print(f"[ERROR] Process failed: {e}. Original files were NOT deleted.")

def export_model_to_onnx(model, config, engine, ckpt_path=None):
    """
    Universal function to export any trained Anomalib model to ONNX format.

    Args:
        model: The initialized and trained Anomalib model object.
        config (dict): The configuration dictionary loaded from config.yaml.
        engine (Engine): The fitted Anomalib Engine instance.
        ckpt_path (str, optional): Path to a specific checkpoint.
    """
    export_dir = config.get("paths", {}).get("exports_onnx_path", "results/exports")

    model_name = model.__class__.__name__
    model_arch = config.get("model_architecture", {})
    backbone = model_arch.get("backbone", "default_backbone")
    gen_config = config.get("general_configuration", {})

    if model_name.lower() == "patchcore" and "efficientnet" in backbone:
        input_size = tuple(gen_config.get("crop_size", [224, 224]))
    else:
        input_size = tuple(gen_config.get("image_size", [256, 256]))

    print(f"\n--- Starting ONNX export for {model_name} ---")
    print(f"Input dimensions expected by the ONNX graph: {input_size}")

    try:
        # Use the already configured engine instead of a new empty one
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=input_size,
            onnx_kwargs={"dynamo": False}
        )

        timestamp = config.get("global_timestamp", "latest")

        # Ensure export_path is treated as a string path before splitting
        export_path_str = str(export_path)
        directory, original_filename = os.path.split(export_path_str)
        _, extension = os.path.splitext(original_filename)

        new_filename = f"{timestamp}_{model_name}_{backbone}{extension}"
        new_export_path = os.path.join(directory, new_filename)

        os.rename(export_path_str, new_export_path)

        print(f"[SUCCESS] Model successfully exported and saved to: {new_export_path}")
        return new_export_path

    except Exception as e:
        print(f"[ERROR] ONNX Export failed: {e}")
        return None


def save_config_file(config, model):
    """
    Helper function that create a copy of the original file configuration as backup in order to ensure test reproducibility
    """
    try:
        model_architecture = config["model_architecture"]
        timestamp = config["gobal_timestamp"]
        backbone = model_architecture["backbone"]
        layers = model_architecture["layers"]
        layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
        config_src_path = Path(config["paths"]["config_src_path"])
        config_dst_dir = Path(config["paths"]["config_dst_path"])

        config_dst_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{timestamp}_{model}_config_{backbone}_{layers_str}.yaml"
        config_dst_path = config_dst_dir / file_name


        print("Saving configruation file...")
        shutil.copy2(config_src_path, config_dst_path)
        print(f"File saved on {config_dst_path}")
    except FileNotFoundError:
        print(f"Error: source configuration file not found at {config_src_path}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    except KeyError as e:
        print(f"Error: Missing key in configuration dictionary: {e}")

def save_prediction_triplet(img_path: str, score: float, anomaly_map, pred_mask, config: dict, datamodule_cfg: dict):
    """
    Reads the original image, processes anomaly maps and masks,
    concatenates them into a triplet, overlays the score, and saves it.
    Dynamically creates routing directories based on the provided configurations.
    """
    model_arch = config.get("model_architecture", {})
    anomaly_images_dir = config.get("anomaly_images_dir", config.get("anomlay_images_dir", "results"))
    dataset_version = config.get("dataset_pipeline", {}).get("dataset_version", "unknown")
    timestamp = config.get("global_timestamp", "000000")
    layers = model_arch.get("layers", ["custom"])
    backbone = model_arch.get("backbone", "custom")

    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    new_folder_name = f"{timestamp}_{backbone}_{layers_str}_d{dataset_version}"

    results_dir = Path(anomaly_images_dir) / new_folder_name
    good_dir = results_dir / "good"
    reject_dir = results_dir / "reject"

    good_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    normal_folder_name = Path(datamodule_cfg.get("test_dir_good", "test/good")).name
    img_path_obj = Path(img_path)

    if normal_folder_name in img_path_obj.parts:
        target_dir = good_dir
    else:
        target_dir = reject_dir

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"[WARNING] Could not read image: {img_path}")
        return

    h, w = img_orig.shape[:2]

    if anomaly_map is not None:
        # Move to CPU, remove extra dimensions, and convert to numpy
        a_map = anomaly_map.squeeze().cpu().numpy()
        # Normalize to 0-255 and apply color map
        a_map_norm = cv2.normalize(a_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(a_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w, h))
    else:
        # Fallback if map is missing
        heatmap = np.zeros_like(img_orig)

    if pred_mask is not None:
        p_mask = pred_mask.squeeze().cpu().numpy()
        p_mask_colored = (p_mask * 255).astype(np.uint8)
        p_mask_colored = cv2.cvtColor(p_mask_colored, cv2.COLOR_GRAY2BGR)
        p_mask_colored = cv2.resize(p_mask_colored, (w, h))
    else:
        p_mask_colored = np.zeros_like(img_orig)

    triplet_img = cv2.hconcat([img_orig, heatmap, p_mask_colored])

    text = f"Anomaly Score: {score:.4f}"
    cv2.putText(
        img=triplet_img,
        text=text,
        org=(20, 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255), # Red text in BGR
        thickness=2,
        lineType=cv2.LINE_AA
    )

    save_path = target_dir / f"{img_path_obj.stem}_triplet.jpg"
    cv2.imwrite(str(save_path), triplet_img)
