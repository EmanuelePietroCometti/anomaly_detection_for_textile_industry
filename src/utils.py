from pathlib import Path
import shutil
import os
from anomalib.engine import Engine
from anomalib.deploy import ExportType

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

def export_model_to_onnx(model, config, ckpt_path=None):
    """
    Universal function to export any trained Anomalib model to ONNX format.
    It dynamically detects the model type to assign the correct input size and nomenclature.
    
    Args:
        model: The initialized and trained Anomalib model object.
        config (dict): The configuration dictionary loaded from config.yaml.
        ckpt_path (str, optional): Path to a specific checkpoint. If None, it exports the current state of the model.
        
    Returns:
        str: The final path of the exported ONNX model, or None if it fails.
    """
    export_dir = config.get("paths", {}).get("exports_onnx_path", "results/exports")
    
    # Dynamically extract model properties
    model_name = model.__class__.__name__
    model_arch = config.get("model_architecture", {})
    backbone = model_arch.get("backbone", "default_backbone")
    
    # Determine the correct input size dynamically
    # Patchcore with EfficientNet relies heavily on the crop_size as the effective input, 
    # while other models like EfficientAD and RD4AD process the full image_size.
    gen_config = config.get("general_configuration", {})
    if model_name.lower() == "patchcore" and "efficientnet" in backbone:
        input_size = tuple(gen_config.get("crop_size", [224, 224]))
    else:
        input_size = tuple(gen_config.get("image_size", [256, 256]))

    engine = Engine()

    print(f"\n--- Starting ONNX export for {model_name} ---")
    print(f"Input dimensions expected by the ONNX graph: {input_size}")
    
    try:
        # Export the model using Anomalib's internal Engine
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=input_size,
            onnx_kwargs={
                "dynamo": False, # Disabled dynamo for robust compatibility across deployment environments
            }
        )
        
        # Rename the exported file for better traceability
        timestamp = config.get("global_timestamp", "latest")
        directory, original_filename = os.path.split(export_path)
        _, extension = os.path.splitext(original_filename)
        
        # Safe formatting: {timestamp}_{ModelName}_{backbone}.onnx
        new_filename = f"{timestamp}_{model_name}_{backbone}{extension}"
        new_export_path = os.path.join(directory, new_filename)
        
        os.rename(export_path, new_export_path)
        
        print(f"[SUCCESS] Model successfully exported and saved to: {new_export_path}")
        return new_export_path
        
    except Exception as e:
        print(f"[ERROR] ONNX Export failed: {e}")
        return None