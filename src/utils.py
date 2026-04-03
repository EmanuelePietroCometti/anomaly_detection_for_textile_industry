from pathlib import Path
import shutil
import os
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import cv2
import numpy as np

def rename_run_and_update_symlink(symlink_path, backbone, layers, config):
    """
    Renames the directory pointed by the symlink and updates the symlink 
    to point to the new folder name, keeping everything in the same location.
    """
    dataset_cfg = config.get("dataset_pipeline", {})
    dataset_version = dataset_cfg.get("dataset_version", "unknown")
    timestamp = config.get("global_timestamp", "000000")
    
    symlink_obj = Path(symlink_path)
    
    if symlink_obj.is_symlink():
        real_source_dir = symlink_obj.resolve()
    elif symlink_obj.is_dir():
        real_source_dir = symlink_obj
    else:
        print(f"[INFO] No valid run directory or symlink found at '{symlink_path}'.")
        return

    if not real_source_dir.exists():
        print(f"[ERROR] The target directory '{real_source_dir}' does not exist.")
        return

    layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)
    new_name = f"{timestamp}_{backbone}_{layers_str}_d{dataset_version}"
    
    new_dir_path = real_source_dir.parent / new_name

    try:
        real_source_dir.rename(new_dir_path)
        print(f"[SUCCESS] Directory renamed to: {new_name}")

        if symlink_obj.is_symlink():
            symlink_obj.unlink()
            symlink_obj.symlink_to(new_dir_path.name) 
            print(f"[INFO] Symlink '{symlink_obj.name}' updated to point to the new folder.")
        return True
    except Exception as e:
        print(f"[ERROR] Rename/Symlink process failed: {e}")
        return False

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
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=input_size,
            onnx_kwargs={"dynamo": False}
        )

        timestamp = config.get("global_timestamp", "latest")

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
    Creates a copy of the original configuration file as backup to ensure test reproducibility.
    """
    try:
        model_architecture = config.get("model_architecture", {})
        timestamp = config.get("global_timestamp", "latest")
        backbone = model_architecture.get("backbone", "unknown")
        layers = model_architecture.get("layers", [])
        layers_str = "_".join(layers) if isinstance(layers, list) else str(layers)

        model_name = model.__class__.__name__

        config_src_path = Path(config["paths"]["config_src_path"])
        config_dst_dir = Path(config["paths"]["config_dst_path"])

        config_dst_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{timestamp}_{model_name}_config_{backbone}_{layers_str}.yaml"
        config_dst_path = config_dst_dir / file_name

        shutil.copy(config_src_path, config_dst_path)
        print(f"[SUCCESS] Config successfully copied to: {config_dst_path}")

    except Exception as e:
        print(f"[ERROR] Failed to copy config file: {e}")

def save_prediction_triplet(img_path: str, score: float, anomaly_map, pred_mask, config: dict, datamodule_cfg: dict):
    """
    Reads the original image, processes anomaly maps and masks,
    concatenates them into a triplet, overlays the score, and saves it.
    """
    model_arch = config.get("model_architecture", {})
    # Correctly extract the anomaly_images path from the config dictionary
    paths_cfg = config.get("paths", {})
    anomaly_images_dir = paths_cfg.get("anomaly_images", "results/anomaly_images")

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

    target_dir = good_dir if normal_folder_name in img_path_obj.parts else reject_dir

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"[WARNING] Could not read image: {img_path}")
        return

    h, w = img_orig.shape[:2]

    if anomaly_map is not None:
        if hasattr(anomaly_map, 'cpu'):
            anomaly_map = anomaly_map.cpu().numpy()
        a_map = anomaly_map.squeeze()
        a_map_norm = cv2.normalize(a_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(a_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (w, h))
    else:
        heatmap = np.zeros_like(img_orig)

    if pred_mask is not None:
        if hasattr(pred_mask, 'cpu'):
            pred_mask = pred_mask.cpu().numpy()
        p_mask = pred_mask.squeeze()
        p_mask_colored = (p_mask * 255).astype(np.uint8)
        p_mask_colored = cv2.cvtColor(p_mask_colored, cv2.COLOR_GRAY2BGR)
        p_mask_colored = cv2.resize(p_mask_colored, (w, h))
    else:
        p_mask_colored = np.zeros_like(img_orig)

    triplet_img = cv2.hconcat([img_orig, heatmap, p_mask_colored])

    text = f"Anomaly Score: {score:.4f}"
    cv2.putText(
        img=triplet_img, text=text, org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA
    )

    save_path = target_dir / f"{img_path_obj.stem}_triplet.jpg"
    cv2.imwrite(str(save_path), triplet_img)


if __name__ == "__main__":
    rename_run_and_update_symlink()