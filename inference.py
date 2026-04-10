import os
import cv2
import argparse
from pathlib import Path
from anomalib.deploy import TorchInferencer

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
    for img_path in image_list:
        # Load image with OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARNING] Skipping {img_path.name}, could not read file.")
            continue
            
        # Convert BGR to RGB as expected by the model pipeline
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predict
        # The output contains: pred_score, pred_label, anomaly_map, and pred_mask
        predictions = inferencer.predict(image=image)

        # Determine status based on the threshold embedded in the .pt file
        status = "REJECT" if predictions.pred_label else "GOOD"
        score = predictions.pred_score

        print(f"Image: {img_path.name} | Status: {status} | Score: {score:.4f}")

        # Save Visualization (Heatmap)
        if predictions.anomaly_map is not None:
            # Normalize anomaly map for visualization
            heatmap = cv2.normalize(predictions.anomaly_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Save the result
            save_name = f"{status}_{score:.3f}_{img_path.name}"
            cv2.imwrite(str(output_path_obj / save_name), heatmap)

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