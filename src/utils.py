from pathlib import Path
import shutil
import os
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import cv2
import numpy as np
from .config import load_config
import glob

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
    
def export_model_to_pt(model, config, engine):
    """
    Universal function to export any trained Anomalib model to TORCH format.

    Args:
        model: The initialized and trained Anomalib model object.
        config (dict): The configuration dictionary loaded from config.yaml.
        engine (Engine): The fitted Anomalib Engine instance.
        ckpt_path (str, optional): Path to a specific checkpoint.
    """
    export_dir = config.get("paths", {}).get("exports_pt_path", "results/exports")
    model_name = model.__class__.__name__
    model_arch = config.get("model_architecture", {})
    backbone = model_arch.get("backbone", "default_backbone")
    gen_config = config.get("general_configuration", {})
    
    if model_name.lower() == "patchcore" and "efficientnet" in backbone:
        input_size = tuple(gen_config.get("crop_size", [224, 224]))
    else:
        input_size = tuple(gen_config.get("image_size", [256, 256]))

    print(f"\n--- Starting TORCH export for {model_name} ---")
    print(f"Input dimensions expected by the TORCH graph: {input_size}")

    try:
        export_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=export_dir,
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
        print(f"[ERROR] TORCH Export failed: {e}")
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
    concatenates them into a triplet [Original | Heatmap Overlay | Segmentation Boundaries], 
    overlays the score, and saves it.
    """
    model_arch = config.get("model_architecture", {})
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

    # Route the image to 'good' or 'reject' folder based on its original path
    target_dir = good_dir if normal_folder_name in img_path_obj.parts else reject_dir

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"[WARNING] Could not read image: {img_path}")
        return

    h, w = img_orig.shape[:2]

    # Prepare heatmap overlay
    overlay_img = img_orig.copy()
    if anomaly_map is not None:
        a_map = anomaly_map.detach().cpu().numpy().squeeze() if hasattr(anomaly_map, 'cpu') else anomaly_map.squeeze()
        
        heatmap_norm = cv2.normalize(a_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        if heatmap_colored.shape[:2] != (h, w):
            heatmap_colored = cv2.resize(heatmap_colored, (w, h))
        
        overlay_img = cv2.addWeighted(img_orig, 0.5, heatmap_colored, 0.5, 0)

    # Prepare segmentation mask
    segmentation_img = img_orig.copy()
    if pred_mask is not None:
        p_mask = pred_mask.detach().cpu().numpy().squeeze() if hasattr(pred_mask, 'cpu') else pred_mask.squeeze()
        
        mask_uint8 = (p_mask * 255).astype(np.uint8) if p_mask.max() <= 1.0 else p_mask.astype(np.uint8)
        
        if mask_uint8.shape[:2] != (h, w):
            mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

        # Extract boundaries of the defect and draw them in red
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(segmentation_img, contours, -1, (0, 0, 255), 2)
        
        # Draw contours on the heatmap overlay for better clarity
        cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)

    # Concatenate and add Score
    triplet_img = np.hstack((img_orig, overlay_img, segmentation_img))

    text = f"Anomaly Score: {score:.4f}"
    cv2.putText(
        img=triplet_img, text=text, org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA
    )
    save_path = target_dir / f"{img_path_obj.stem}_triplet.jpg"
    cv2.imwrite(str(save_path), triplet_img)

def convert_masks(input_dir: str, output_dir: str):
    """
    Converts low_intenisty mask images from MVTEC DL Tool into visibile binary masks and save them in BMP format.
    """
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()

    mask_paths = []
    valid_extensions = config.get("general_configuration", {}).get("valid_extensions", [])

    for ext in valid_extensions:
        search_pattern = os.path.join(input_dir, f"*{ext}")
        mask_paths.extend(glob.glob(search_pattern))
    
    if not mask_paths:
        print(f"[WARNING] No mask files found in '{input_dir}' with specified extensions.")
        return
    
    for path in mask_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARNING] Could not read mask image: {path}")
            continue

        visible_mask = np.where(img > 0, 255, 0).astype(np.uint8)
        filename = os.path.basename(path)
        filename_no_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{filename_no_ext}.bmp")
        cv2.imwrite(out_path, visible_mask)
        print(f"Processed {filename} -> {os.path.basename(out_path)}")

if __name__ == "__main__":
    #rename_run_and_update_symlink()
    convert_masks(input_dir="data/masks_raw", output_dir="data/masks_converted")