import os
import warnings
import cv2
import argparse
from pathlib import Path
import numpy as np
from anomalib.deploy import TorchInferencer

os.environ["TRUST_REMOTE_CODE"] = "1"
warnings.filterwarnings("ignore", message=".*TorchInferencer is a legacy inferencer.*")

def run_inference(model_path: str, input_path: str, output_root: str, device: str = "cpu"):
    """
    Standalone inference module to process images using an exported Anomalib .pt model.
    """
    # Initialize the Inferencer
    # The .pt file contains the model weights, architecture, and required transforms.
    try:
        inferencer = TorchInferencer(path=model_path, device=device)
        print(f"[INFO] Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Setup Paths
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_root)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Filter for valid image extensions based on your config
    valid_extensions = {".bmp", ".png", ".jpg", ".jpeg"}
    
    if input_path_obj.is_file():
        image_list = [input_path_obj]
    else:
        image_list = [p for p in input_path_obj.iterdir() if p.suffix.lower() in valid_extensions]

    print(f"[INFO] Processing {len(image_list)} images...")

    # Inference Loop
    # Inference Loop
    for img_path in image_list:
        # Load image with OpenCV in BGR format (Standard for OpenCV saving)
        original_bgr = cv2.imread(str(img_path))
        if original_bgr is None:
            print(f"[WARNING] Skipping {img_path.name}, could not read file.")
            continue
            
        # Convert BGR to RGB for the model prediction
        image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # Predict using the loaded TorchInferencer
        predictions = inferencer.predict(image=image_rgb)

        # Determine status and safely extract the score
        status = "REJECT" if predictions.pred_label else "GOOD"
        raw_score = predictions.pred_score
        score = raw_score.item() if hasattr(raw_score, 'item') else float(raw_score)

        print(f"Image: {img_path.name} | Status: {status} | Score: {score:.4f}")

        if predictions.anomaly_map is not None:
            
            # Prepare the Heatmap Overlay
            a_map = predictions.anomaly_map
            a_map = a_map.detach().cpu().numpy().squeeze() if hasattr(a_map, 'cpu') else a_map.squeeze()
            
            # Create the colored heatmap
            heatmap_norm = cv2.normalize(a_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # Ensure dimensions match before overlaying
            if heatmap_colored.shape[:2] != original_bgr.shape[:2]:
                heatmap_colored = cv2.resize(heatmap_colored, (original_bgr.shape[1], original_bgr.shape[0]))
                
            # Superimpose the heatmap onto the original image (50% transparency)
            overlay_img = cv2.addWeighted(original_bgr, 0.5, heatmap_colored, 0.5, 0)

            # Prepare the Segmentation Mask (Contours)
            segmentation_img = original_bgr.copy()
            
            if predictions.pred_mask is not None:
                p_mask = predictions.pred_mask
                p_mask = p_mask.detach().cpu().numpy().squeeze() if hasattr(p_mask, 'cpu') else p_mask.squeeze()
                
                # Convert binary mask to 0-255 uint8 format
                mask_uint8 = (p_mask * 255).astype(np.uint8) if p_mask.max() <= 1.0 else p_mask.astype(np.uint8)
                
                if mask_uint8.shape[:2] != original_bgr.shape[:2]:
                    mask_uint8 = cv2.resize(mask_uint8, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Extract boundaries of the defect and draw them in red
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(segmentation_img, contours, -1, (0, 0, 255), 2)
                
                # Optional: Also draw contours on the heatmap overlay for better clarity
                cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)

            # Concatenate horizontally to create the Anomalib-style grid 
            # [Original | Heatmap Overlay | Segmentation Boundaries]
            combined_result = np.hstack((original_bgr, overlay_img, segmentation_img))
            
            # Save the final concatenated image
            save_name = f"{status}_{score:.3f}_{img_path.name}"
            cv2.imwrite(str(output_path_obj / save_name), combined_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Anomalib Inference Module")
    parser.add_argument("--weights", type=str, required=True, help="Path to the exported model.pt file")
    parser.add_argument("--input", type=str, required=True, help="Path to an image or a directory of images")
    parser.add_argument("--output", type=str, default="results/inference_results", help="Directory to save output heatmaps")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run inference on")

    args = parser.parse_args()
    
    run_inference(
        model_path=args.weights,
        input_path=args.input,
        output_root=args.output,
        device=args.device
    )