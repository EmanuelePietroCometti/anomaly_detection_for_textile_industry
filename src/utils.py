from pathlib import Path
import shutil
import os
import torch
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import cv2
import numpy as np
from .config import load_config
import glob
import albumentations as A
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.transforms import functional as F_v2
from lightning.pytorch.callbacks import Callback

class GPUAugmentationCallback(Callback):
    """
    PyTorch Lightning Callback executing pure-GPU advanced augmentations
    specifically designed to make models robust against dust noise on textile datasets.
    """
    def __init__(self, crop_padding=30, equalization_p=0.5):
        super().__init__()
        self.padding = crop_padding
        self.equalization_p = equalization_p
        
        # Optimized pipeline for textile defect detection ignoring dust
        self.train_transforms = v2.Compose([
            # Spatial/Geometry: Standard textile rotations and slight scales
            v2.RandomAffine(
                degrees=[-10.0, 10.0],      
                translate=[0.02, 0.02],   
                scale=[0.98, 1.02],       
                fill=1.0, # White background padding
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            # Color Agnosticism: Heavy jittering so the network stops relying
            # on precise color signatures of dust vs straw
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            v2.RandomGrayscale(p=0.2),
            
            # High-Frequency Mitigation: Blurring forces the network to look at 
            # macro textile structures (straws) rather than microscopic specs (dust)
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.4)
        ])

    def _apply_gpu_equalization(self, images):
        """
        Applies histogram equalization on a per-image basis within the batch.
        Handles float32 [0.0, 1.0] -> uint8 [0, 255] -> float32 [0.0, 1.0] conversion on GPU.
        """
        equalized_images = []

        for i in range(images.shape[0]):
            img = images[i]

            # Denormalize to uint8 for torchvision functional compatibility
            img_unint8 = (img * 255.0).clamp(0, 255).to(torch.uint8)

            # Appòly GPU equalization using torchvision's functional API
            img_eq = F_v2.equalize(img_unint8)

            # Bring back to float32 [0.0, 1.0]
            img_eq_float = img_eq.to(torch.float32) / 255.0
            equalized_images.append(img_eq_float)

        return torch.stack(equalized_images)
    

    def _apply_dynamic_gpu_crop(self, images, masks=None):
        """Translates OpenCV Dynamic Crop into pure PyTorch Tensor operations."""
        B, C, H, W = images.shape
        cropped_imgs = []
        cropped_masks = [] if masks is not None else None
        
        # Assuming background is close to 1.0 (white normalized)
        gray = images.mean(dim=1) 
        is_dark = gray < 0.94 
        
        for i in range(B):
            img = images[i]
            mask_i = is_dark[i]
            coords = torch.nonzero(mask_i)
            
            if coords.numel() == 0:
                cropped_imgs.append(img)
                if masks is not None: cropped_masks.append(masks[i])
                continue
            
            y_min, x_min = coords.min(dim=0).values
            y_max, x_max = coords.max(dim=0).values
            
            h_box = y_max - y_min
            w_box = x_max - x_min
            size = torch.maximum(h_box, w_box)
            
            center_y = y_min + h_box // 2
            center_x = x_min + w_box // 2
            
            y1 = torch.clamp(center_y - size // 2 - self.padding, min=0)
            y2 = torch.clamp(center_y + size // 2 + self.padding, max=H)
            x1 = torch.clamp(center_x - size // 2 - self.padding, min=0)
            x2 = torch.clamp(center_x + size // 2 + self.padding, max=W)
            
            crop_img = img[:, y1:y2, x1:x2].unsqueeze(0) 
            resized_img = F_v2.resize(
                crop_img, size=[H, W], 
                interpolation=v2.InterpolationMode.BILINEAR, antialias=True
            )
            cropped_imgs.append(resized_img.squeeze(0))
            
            if masks is not None:
                crop_mask = masks[i:i+1, y1:y2, x1:x2].unsqueeze(0)
                resized_mask = F_v2.resize(
                    crop_mask, size=[H, W], 
                    interpolation=v2.InterpolationMode.NEAREST
                )
                cropped_masks.append(resized_mask.squeeze(0).squeeze(0))

        res_imgs = torch.stack(cropped_imgs)
        res_masks = torch.stack(cropped_masks) if masks is not None else None
        return res_imgs, res_masks

    def _process_batch(self, batch, is_train=False):
        """Processes batches directly on GPU executing tailored noise injection."""
        images = batch.image if hasattr(batch, 'image') else batch['image']
        
        masks = None
        if hasattr(batch, 'mask'):
            masks = batch.mask
        elif isinstance(batch, dict) and 'mask' in batch:
            masks = batch['mask']

        # Dynamic cropping (always active)
        images, masks = self._apply_dynamic_gpu_crop(images, masks)

        # Augmentations logic
        if is_train:
            if torch.rand(1, device = images.device).item() < self.equalization_p:
                images = self._apply_gpu_equalization(images)

            images = self.train_transforms(images)
            
            # Dynamic Dust Simulation (Salt & Pepper Noise)
            # Injecting artificial dust teaches the model that tiny black/white dots 
            # do not change the classification target.
            if torch.rand(1, device=images.device).item() < 0.4:
                # Generate a random binary mask for dust coordinates
                dust_mask_black = torch.rand_like(images[:, 0:1, :, :]) < 0.005 # 0.5% black dust
                dust_mask_white = torch.rand_like(images[:, 0:1, :, :]) < 0.005 # 0.5% white dust
                
                # Apply dust directly to all channels
                images = torch.where(dust_mask_black, torch.zeros_like(images), images)
                images = torch.where(dust_mask_white, torch.ones_like(images), images)

        # Re-assign back to batch
        if hasattr(batch, 'image'):
            batch.image = images
            if masks is not None: batch.mask = masks
        else:
            batch['image'] = images
            if masks is not None: batch['mask'] = masks

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._process_batch(batch, is_train=True)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._process_batch(batch, is_train=False)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._process_batch(batch, is_train=False)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._process_batch(batch, is_train=False)

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
    Universal function to export trained Anomalib (>=2.2.0) models to ONNX format.
    Optimized for PyTorch 2.11.0 and ONNXRuntime-GPU >=1.27.0.

    Args:
        model: The initialized and trained Anomalib model object.
        config (dict): The configuration dictionary loaded from config.yaml.
        engine (Engine): The fitted Anomalib Engine instance.
        ckpt_path (str, optional): Path to a specific checkpoint.
    """
    # Safely extract paths and configurations
    export_dir = config.get("paths", {}).get("exports_onnx_path", "results/exports")
    model_name = model.__class__.__name__
    model_arch = config.get("model_architecture", {})
    backbone = model_arch.get("backbone", "default_backbone")
    gen_config = config.get("general_configuration", {})

    # Define input dimensions based on the model type
    if model_name.lower() == "patchcore" and "efficientnet" in backbone:
        input_size = tuple(gen_config.get("crop_size", [224, 224]))
    else:
        input_size = tuple(gen_config.get("image_size", [256, 256]))

    print(f"\n--- Starting ONNX export for {model_name} ---")
    print(f"Input dimensions expected by the ONNX graph: {input_size}")

    # Explicitly map the batch dimension (index 0) to a dynamic string ('batch_size')
    dynamic_batch_config = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    try:
        # Anomalib 2.2.0+ engine.export wrapper
        export_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=export_dir,
            ckpt_path=ckpt_path,
            input_size=input_size,
            # Kwargs passed directly to torch.onnx.export
            onnx_kwargs={
                "dynamo": False,               # Force TorchScript exporter (required for dynamic_axes dict)
                "opset_version": 17,           # Recommended opset for ONNX >=1.21.0 and Torch 2.11
                "input_names": ["input"],      # Bind input name to dynamic_axes key
                "output_names": ["output"],    # Bind output name to dynamic_axes key
                "dynamic_axes": dynamic_batch_config
            }
        )

        # Handle file renaming post-export
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
    """
    export_dir = config.get("paths", {}).get("exports_pt_path", "results/exports")
    model_name = model.__class__.__name__
    model_arch = config.get("model_architecture", {})
    backbone = model_arch.get("backbone", "default_backbone")
    gen_config = config.get("general_configuration", {})
    
    # Define input dimensions based on the model type
    if model_name.lower() == "patchcore" and "efficientnet" in backbone:
        input_size = tuple(gen_config.get("crop_size", [224, 224]))
    else:
        input_size = tuple(gen_config.get("image_size", [256, 256]))

    print(f"\n--- Starting TORCH export for {model_name} ---")
    print(f"Input dimensions expected by the TORCH graph: {input_size}")

    try:
        # Anomalib 2.2.0+ engine.export wrapper for TorchScript
        export_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=export_dir,
            input_size=input_size
        )

        # Handle file renaming post-export
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
    convert_masks(input_dir="D:\\emanuele\\Code\\SuperSimpleNet\\dataset\\active_pool\\masks", output_dir="D:\\emanuele\\Code\\SuperSimpleNet\\dataset\\active_pool\\masks")